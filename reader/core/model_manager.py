"""Менеджер модели XTTS-v2: загрузка, устройство (CPU/GPU), прогрев, латенты.

Содержит совместимость-патчи:
  • torchaudio.load → soundfile (обход TorchCodec/FFmpeg на Windows);
  • safe globals для torch.load с weights_only=True в PyTorch 2.6+.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

import numpy as np
import torch

from reader.core import config, voices

logger = logging.getLogger("reader.model")

# ------------------------------------------------------------------ Патч torchaudio
def install_soundfile_audio_loader() -> None:
    """Заменяет torchaudio.load на чтение через soundfile.

    Зачем: torchaudio ≥ 2.6 грузит аудио через TorchCodec, который на Windows
    требует FFmpeg «full-shared» с DLL. soundfile работает без FFmpeg и читает
    WAV/FLAC/OGG из коробки.
    """
    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        logger.warning("soundfile не установлен — не могу пропатчить torchaudio.load")
        return

    try:
        import torchaudio  # type: ignore
    except ImportError:
        logger.warning("torchaudio не установлен — пропускаю патч")
        return

    if getattr(torchaudio.load, "_xtts_soundfile_patch", False):
        return

    _orig_load = torchaudio.load

    def _sf_load(uri, *args, **kwargs):
        try:
            if isinstance(uri, (str, os.PathLike)):
                path = os.fspath(uri)
            else:
                return _orig_load(uri, *args, **kwargs)
        except Exception:
            return _orig_load(uri, *args, **kwargs)

        try:
            data, sr = sf.read(path, dtype="float32", always_2d=True)
        except Exception:
            return _orig_load(uri, *args, **kwargs)

        tensor = torch.from_numpy(np.ascontiguousarray(data.T))
        return tensor, int(sr)

    _sf_load._xtts_soundfile_patch = True  # type: ignore[attr-defined]
    torchaudio.load = _sf_load  # type: ignore[assignment]
    logger.info("torchaudio.load заменён на soundfile-загрузчик (обходим TorchCodec/FFmpeg)")


install_soundfile_audio_loader()


# ------------------------------------------------------------------ Safe globals
def register_xtts_safe_globals() -> None:
    """PyTorch 2.6+ по умолчанию использует weights_only=True.

    XTTS-чекпоинт содержит конфиг-классы, которые нужно явно разрешить,
    иначе torch.load падает с UnpicklingError.
    """
    try:
        from TTS.tts.configs.xtts_config import XttsConfig  # type: ignore
        safe: list[Any] = [XttsConfig]
        try:
            from TTS.tts.models.xtts import XttsAudioConfig, XttsArgs  # type: ignore
            safe += [XttsAudioConfig, XttsArgs]
        except Exception:
            pass
        try:
            from TTS.config.shared_configs import BaseDatasetConfig  # type: ignore
            safe.append(BaseDatasetConfig)
        except Exception:
            pass

        torch.serialization.add_safe_globals(safe)
        logger.info("Зарегистрированы safe globals для XTTS: %s",
                    [c.__name__ for c in safe])
    except Exception as e:
        logger.warning("Не удалось зарегистрировать safe globals: %s", e)


# ------------------------------------------------------------------ CUDA
def cuda_info() -> dict:
    info = {"available": False, "name": "", "vram_gb": 0.0}
    try:
        if not torch.cuda.is_available():
            return info
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        info["available"] = True
        info["name"] = props.name
        info["vram_gb"] = props.total_memory / (1024 ** 3)
    except Exception:
        logger.exception("Не удалось получить информацию о CUDA")
    return info


def default_use_gpu(cuda: dict) -> bool:
    if not cuda["available"]:
        return False
    return cuda["vram_gb"] >= config.FP16_VRAM_THRESHOLD_GB


# ------------------------------------------------------------------ Менеджер
class ModelManager:
    """Обёртка над TTS-объектом: устройство, кэш латентов, инференс."""

    def __init__(self) -> None:
        self.tts: Any | None = None
        self.sample_rate: int = 24000
        self.char_limit: int = config.DEFAULT_CHAR_LIMIT
        self.chunk_max: int = max(
            config.MIN_CHUNK_LEN,
            config.DEFAULT_CHAR_LIMIT - config.CHUNK_SAFETY_MARGIN,
        )
        self.device: str = "cpu"
        self.cuda: dict = cuda_info()
        self.latents_cache: dict[tuple[str, str], tuple[Any, Any]] = {}

    # ------------------------------------------------ properties
    @property
    def ready(self) -> bool:
        return self.tts is not None

    @property
    def xtts_model(self):
        synth = getattr(self.tts, "synthesizer", None)
        return getattr(synth, "tts_model", None) if synth else None

    # ------------------------------------------------ loading
    def load(self, want_gpu: bool, status_cb=None) -> None:
        """Синхронная загрузка модели (вызывать из рабочего потока)."""
        emit = status_cb or (lambda _s: None)
        register_xtts_safe_globals()

        logger.info("Загрузка модели %s …", config.MODEL_NAME)
        from TTS.api import TTS  # type: ignore

        want_gpu = want_gpu and self.cuda["available"]
        initial_device = "cuda" if want_gpu else "cpu"
        emit(f"⏳ Загрузка XTTS-v2 на {initial_device.upper()}…")

        self.tts = TTS(config.MODEL_NAME)
        self.tts.to("cpu")

        try:
            model = self.xtts_model
            if model is not None:
                model.eval()
        except Exception:
            pass

        synth = getattr(self.tts, "synthesizer", None)
        sr = getattr(synth, "output_sample_rate", None) if synth else None
        self.sample_rate = int(sr) if sr else 24000

        self._detect_char_limits(synth)

        self.device = "cpu"
        if initial_device == "cuda":
            if self.try_move_to_cuda():
                self.device = "cuda"
            else:
                emit("⚠ Не удалось использовать GPU, работаем на CPU")

        emit("🔥 Прогрев модели…")
        self.warmup()

        logger.info(
            "Модель готова: устройство=%s, sr=%d Гц, char_limit=%d, chunk_max=%d",
            self.device.upper(), self.sample_rate, self.char_limit, self.chunk_max,
        )

    def _detect_char_limits(self, synth: Any) -> None:
        char_limit = config.DEFAULT_CHAR_LIMIT
        try:
            model = getattr(synth, "tts_model", None)
            tokenizer = getattr(model, "tokenizer", None) if model else None
            limits = getattr(tokenizer, "char_limits", None) if tokenizer else None
            if isinstance(limits, dict):
                found = limits.get(config.LANGUAGE)
                if isinstance(found, int) and found > 0:
                    char_limit = found
        except Exception:
            logger.exception("Не удалось определить char_limit, беру дефолт")
        self.char_limit = char_limit
        self.chunk_max = max(config.MIN_CHUNK_LEN,
                             char_limit - config.CHUNK_SAFETY_MARGIN)

    # ------------------------------------------------ device switching
    def switch_device(self, target: str, status_cb=None) -> bool:
        emit = status_cb or (lambda _s: None)
        logger.info("Переключение устройства: %s → %s",
                    self.device.upper(), target.upper())
        t0 = time.time()
        if target == "cuda":
            if not self.try_move_to_cuda():
                return False
        else:
            self.move_to_cpu()

        emit(f"🔥 Прогрев на {target.upper()}…")
        self.warmup()
        logger.info("Переключение завершено за %.1f с", time.time() - t0)
        return True

    def try_move_to_cuda(self) -> bool:
        assert self.tts is not None
        try:
            vram = self.cuda["vram_gb"]
            use_half = vram < config.FP16_VRAM_THRESHOLD_GB
            if use_half:
                logger.info("VRAM=%.1f ГБ < %.1f ГБ — используем FP16",
                            vram, config.FP16_VRAM_THRESHOLD_GB)

            model = self.xtts_model
            try:
                if use_half:
                    self.tts.to("cuda", dtype=torch.float16)
                else:
                    self.tts.to("cuda")
            except TypeError:
                if model is not None and use_half:
                    model.half()
                self.tts.to("cuda")

            if model is not None:
                first_p = next(model.parameters(), None)
                if first_p is not None:
                    logger.info("GPU активна: %s | dtype=%s",
                                first_p.device, first_p.dtype)
            self.device = "cuda"
            self.latents_cache.clear()
            return True
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda" in msg:
                logger.warning("CUDA OOM или ошибка CUDA: %s", e)
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                self.move_to_cpu()
                return False
            logger.exception("Неожиданная RuntimeError при переносе на GPU")
            return False
        except Exception:
            logger.exception("Не удалось перенести модель на GPU")
            return False

    def move_to_cpu(self) -> None:
        assert self.tts is not None
        try:
            model = self.xtts_model
            if model is not None:
                try:
                    model.float()
                except Exception:
                    pass
            self.tts.to("cpu")
            if self.cuda["available"]:
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            self.device = "cpu"
            self.latents_cache.clear()
            logger.info("Модель перенесена на CPU")
        except Exception:
            logger.exception("Не удалось перенести модель на CPU")

    # ------------------------------------------------ warmup
    def warmup(self) -> None:
        """Прогрев модели на первом попавшемся голосе."""
        try:
            model = self.xtts_model
            if model is None:
                return

            wavs: list = []
            for d in sorted(config.VOICES_DIR.iterdir()):
                if d.is_dir() and d.name != ".cache":
                    wavs = voices.voice_wavs(d)
                    if wavs:
                        break
            if not wavs:
                logger.info("Прогрев пропущен: нет .wav для латентов")
                return

            voice = str(wavs[0])
            logger.info("Прогрев модели на %s (голос: %s)…",
                        self.device.upper(), os.path.basename(voice))
            t0 = time.time()
            gpt, spk = model.get_conditioning_latents(audio_path=voice)
            with torch.inference_mode():
                _ = model.inference(
                    text="Прогрев, раз, два, три.",
                    language=config.LANGUAGE,
                    gpt_cond_latent=gpt,
                    speaker_embedding=spk,
                    **config.XTTS_KWARGS,
                )
            logger.info("Прогрев завершён за %.1f с", time.time() - t0)
        except Exception:
            logger.exception("Прогрев не удался (не критично)")

    # ------------------------------------------------ latents
    def get_latents(self, voice_dir, status_cb=None):
        """Латенты голоса: память → диск-кэш → пересчёт. Возвращает (gpt, spk)."""
        emit = status_cb or (lambda _s: None)
        assert self.tts is not None
        key = (str(voice_dir), self.device)
        cached = self.latents_cache.get(key)
        if cached is not None:
            logger.debug("Латенты из памяти: %s", voice_dir.name)
            return cached

        cache_path = voices.cache_file_for(voice_dir)
        if cache_path.exists():
            try:
                logger.info("Латенты из диска: %s", cache_path.name)
                data = torch.load(cache_path, map_location="cpu", weights_only=False)
                gpt = data["gpt_cond"].to(self.device)
                spk = data["spk_emb"].to(self.device)
                pair = (gpt, spk)
                self.latents_cache[key] = pair
                return pair
            except Exception:
                logger.exception("Кэш повреждён, пересчитываем: %s", cache_path)
                try:
                    cache_path.unlink()
                except Exception:
                    pass

        emit("🧬 Подготовка голоса…")
        wavs = voices.voice_wavs(voice_dir)
        if not wavs:
            raise RuntimeError(f"В папке «{voice_dir.name}» нет .wav")
        logger.info("Расчёт латентов: %s (%d файлов, на %s)",
                    voice_dir.name, len(wavs), self.device.upper())

        model = self.xtts_model
        if model is None:
            raise RuntimeError("TTS-модель не инициализирована (synthesizer is None)")

        gpt_cond, spk_emb = model.get_conditioning_latents(
            audio_path=[str(p) for p in wavs]
        )

        try:
            torch.save(
                {"gpt_cond": gpt_cond.detach().cpu(),
                 "spk_emb": spk_emb.detach().cpu()},
                cache_path,
            )
            logger.info("Латенты сохранены в кэш: %s", cache_path.name)
        except Exception:
            logger.exception("Не удалось сохранить кэш латентов")

        gpt_cond = gpt_cond.to(self.device)
        spk_emb = spk_emb.to(self.device)
        self.latents_cache[key] = (gpt_cond, spk_emb)
        return gpt_cond, spk_emb

    def train_voice(self, voice_dir, status_cb=None) -> float:
        """Пересчёт латентов по всем .wav и сохранение в кэш. Возвращает время в с."""
        emit = status_cb or (lambda _s: None)
        wavs = voices.voice_wavs(voice_dir)
        if not wavs:
            raise RuntimeError(f"В папке «{voice_dir.name}» нет .wav")

        logger.info("Обучение голоса «%s» (%d файлов) на %s",
                    voice_dir.name, len(wavs), self.device.upper())
        emit(f"🧬 Обучение «{voice_dir.name}» ({len(wavs)} записей)…")
        t0 = time.time()

        model = self.xtts_model
        if model is None:
            raise RuntimeError("Модель не инициализирована")

        paths = [str(p) for p in wavs]
        gpt_cond, spk_emb = model.get_conditioning_latents(audio_path=paths)

        cache_path = voices.cache_file_for(voice_dir)
        try:
            torch.save(
                {"gpt_cond": gpt_cond.detach().cpu(),
                 "spk_emb": spk_emb.detach().cpu()},
                cache_path,
            )
            logger.info("Кэш сохранён: %s", cache_path.name)
        except Exception:
            logger.exception("Не удалось сохранить кэш латентов")

        dt = time.time() - t0
        logger.info("Обучение «%s» завершено за %.1f с", voice_dir.name, dt)
        return dt

    # ------------------------------------------------ inference
    def synthesize(self, text: str, gpt_cond, spk_emb) -> np.ndarray:
        """Чистый инференс одного фрагмента без постобработки."""
        model = self.xtts_model
        if model is None:
            raise RuntimeError("TTS-модель не инициализирована")

        if len(text) > self.char_limit:
            cut = text[: self.chunk_max]
            idx = cut.rfind(" ")
            if idx >= self.chunk_max * 0.5:
                cut = cut[:idx].rstrip()
            logger.warning("Обрезан слишком длинный фрагмент (%d → %d симв.)",
                           len(text), len(cut))
            text = cut

        with torch.inference_mode():
            out = model.inference(
                text=text,
                language=config.LANGUAGE,
                gpt_cond_latent=gpt_cond,
                speaker_embedding=spk_emb,
                **config.XTTS_KWARGS,
            )
        return np.asarray(out["wav"], dtype=np.float32).reshape(-1)
