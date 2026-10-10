"""
Русский TTS с клонированием голоса на Coqui XTTS-v2. PyQt6-версия.

Голоса — папки в D:\\Reader\\voices\\:
  voices/
    Alice/            # один голос = одна папка
      sample_01.wav
      sample_02.wav
    Bob/
      bob_clean.wav

Кнопки:
  🎓 Обучить голос   — пересчитать латенты по всем *.wav в папке, сохранить в кэш.
  🎧 Озвучить в файл — использовать кэш (или посчитать и синтезировать).

Зависимости:
  pip install razdel             # токенизация предложений (русский)
  pip install pysbd              # опциональный фолбэк
  pip install lameenc            # для сохранения MP3

Совместимость:
  • PyTorch 2.6+ / torchaudio 2.6+: подменяем torchaudio.load на soundfile-загрузчик,
    чтобы не зависеть от TorchCodec / FFmpeg DLL.
  • PyTorch 2.6+ (weights_only=True по умолчанию) — для XTTS регистрируем safe globals.
  • Корректная нарезка текста: razdel → pysbd → regex.
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import sys
import time
import warnings
import wave
from pathlib import Path
from typing import Any, Optional

# ------------------------------------------------------------------ CPU-потоки
# ВАЖНО: выставляем ДО import torch, иначе MKL их игнорирует.
_CPU_COUNT = os.cpu_count() or 4
for _env in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env, str(_CPU_COUNT))

import numpy as np
import sounddevice as sd
import torch

try:
    torch.set_num_threads(_CPU_COUNT)
except Exception:
    pass
try:
    torch.set_num_interop_threads(1)
except Exception:
    pass
try:
    torch.set_flush_denormal(True)
except Exception:
    pass
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

try:
    from scipy import signal as sps
except Exception:
    sps = None  # type: ignore

try:
    from ruaccent import RUAccent  # type: ignore
    _HAS_RUACCENT = True
except Exception:
    RUAccent = None  # type: ignore[assignment]
    _HAS_RUACCENT = False

# ------------------------------------------------------------------ Sentence splitters
# razdel — лучший для русского: понимает «т.д.», «г.», «т.п.», кавычки-ёлочки, тире.
try:
    from razdel import sentenize as _razdel_sentenize  # type: ignore
    _HAS_RAZDEL = True
except Exception:
    _razdel_sentenize = None  # type: ignore
    _HAS_RAZDEL = False

# pysbd — фолбэк, тоже неплох для русского
try:
    import pysbd as _pysbd  # type: ignore
    _PY_SBD_SEG = _pysbd.Segmenter(language="ru", clean=False)
    _HAS_PYSBD = True
except Exception:
    _PY_SBD_SEG = None
    _HAS_PYSBD = False

from PyQt6.QtCore import QThread, pyqtSignal, Qt, QUrl
from PyQt6.QtGui import QCloseEvent, QDesktopServices, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# ------------------------------------------------------------------ Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reader")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")
warnings.filterwarnings("ignore", message=r".*torch\.jit\.script.*")
warnings.filterwarnings("ignore", message=r".*isin_mps_friendly.*")
warnings.filterwarnings("ignore", message=r".*smallest subnormal.*")


# ------------------------------------------------------------------ Config
LANGUAGE = "ru"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

BASE_DIR = Path(__file__).resolve().parent
VOICES_DIR = BASE_DIR / "voices"
CACHE_DIR = VOICES_DIR / ".cache"
OUTPUT_DIR = BASE_DIR / "output"

for _d in (VOICES_DIR, CACHE_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

XTTS_KWARGS = dict(
    temperature=0.75,
    length_penalty=1.0,
    repetition_penalty=5.0,
    top_k=50,
    top_p=0.85,
    enable_text_splitting=False,
)

DEFAULT_CHAR_LIMIT = 182
CHUNK_SAFETY_MARGIN = 10   # оставляем запас до жёсткого лимита XTTS
MIN_CHUNK_LEN = 40
INSERT_PAUSE_MS = 150      # пауза между чанками, мс

HP_FREQ = 70.0
LP_FREQ = 7500.0
FADE_MS = 15
GATE_THRESH_DB = -50.0

MP3_BITRATE = 192
FP16_VRAM_THRESHOLD_GB = 3.0

STRESS_MARK = "\u0301"


# ============================================================ ПАТЧ torchaudio
def _install_soundfile_audio_loader() -> None:
    """Заменяет torchaudio.load на чтение через soundfile.

    Зачем: torchaudio ≥ 2.6 грузит аудио через TorchCodec, который на Windows
    требует FFmpeg «full-shared» с DLL. Если FFmpeg нет, XTTS не может
    прочитать .wav: 'Could not load libtorchcodec'. soundfile — уже
    зависимость TTS — работает без FFmpeg и читает WAV/FLAC/OGG из коробки.
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
    logger.info("torchaudio.load заменён на soundfile-загрузчик "
                "(обходим TorchCodec/FFmpeg)")


_install_soundfile_audio_loader()


# ------------------------------------------------------------------ Safe globals
def _register_xtts_safe_globals() -> None:
    """PyTorch 2.6+ по умолчанию использует weights_only=True.

    XTTS-чекпоинт содержит конфиг-классы, которые нужно явно
    разрешить, иначе torch.load падает с UnpicklingError.
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


# ------------------------------------------------------ CUDA detection
def _cuda_info() -> dict:
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


_CUDA = _cuda_info()


def _default_use_gpu() -> bool:
    if not _CUDA["available"]:
        return False
    return _CUDA["vram_gb"] >= FP16_VRAM_THRESHOLD_GB


# =========================================================== Module helpers
def _detect_char_limits(synth: Any) -> tuple[int, int]:
    char_limit = DEFAULT_CHAR_LIMIT
    try:
        xtts_model = getattr(synth, "tts_model", None)
        tokenizer = getattr(xtts_model, "tokenizer", None) if xtts_model else None
        limits = getattr(tokenizer, "char_limits", None) if tokenizer else None
        if isinstance(limits, dict):
            found = limits.get(LANGUAGE)
            if isinstance(found, int) and found > 0:
                char_limit = found
    except Exception:
        logger.exception("Не удалось определить char_limit, беру дефолт")
    chunk_max = max(MIN_CHUNK_LEN, char_limit - CHUNK_SAFETY_MARGIN)
    return char_limit, chunk_max


def _try_move_to_cuda(tts: Any) -> bool:
    try:
        vram = _CUDA["vram_gb"]
        use_half = vram < FP16_VRAM_THRESHOLD_GB
        if use_half:
            logger.info("VRAM=%.1f ГБ < %.1f ГБ — используем FP16",
                        vram, FP16_VRAM_THRESHOLD_GB)

        synth = getattr(tts, "synthesizer", None)
        model = getattr(synth, "tts_model", None) if synth else None

        try:
            if use_half:
                tts.to("cuda", dtype=torch.float16)
            else:
                tts.to("cuda")
        except TypeError:
            if model is not None and use_half:
                model.half()
            tts.to("cuda")

        if model is not None:
            first_p = next(model.parameters(), None)
            if first_p is not None:
                logger.info("GPU активна: %s | dtype=%s",
                            first_p.device, first_p.dtype)
        return True
    except RuntimeError as e:
        msg = str(e).lower()
        if "out of memory" in msg or "cuda" in msg:
            logger.warning("CUDA OOM или ошибка CUDA: %s", e)
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                _move_to_cpu(tts)
            except Exception:
                logger.exception("Не удалось откатить модель на CPU")
            return False
        logger.exception("Неожиданная RuntimeError при переносе на GPU")
        return False
    except Exception:
        logger.exception("Не удалось перенести модель на GPU")
        return False


def _move_to_cpu(tts: Any) -> None:
    try:
        synth = getattr(tts, "synthesizer", None)
        model = getattr(synth, "tts_model", None) if synth else None
        if model is not None:
            try:
                model.float()
            except Exception:
                pass
        tts.to("cpu")
        if _CUDA["available"]:
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        logger.info("Модель перенесена на CPU")
    except Exception:
        logger.exception("Не удалось перенести модель на CPU")


def _warmup(tts: Any, device: str) -> None:
    """Прогрев модели на первом попавшемся голосе."""
    try:
        synth = getattr(tts, "synthesizer", None)
        if synth is None:
            return
        xtts_model = getattr(synth, "tts_model", None)
        if xtts_model is None:
            return

        wavs: list[Path] = []
        for d in sorted(VOICES_DIR.iterdir()):
            if d.is_dir() and d.name != ".cache":
                wavs = sorted(d.glob("*.wav"))
                if wavs:
                    break
        if not wavs:
            logger.info("Прогрев пропущен: нет .wav для латентов")
            return

        voice = str(wavs[0])
        logger.info("Прогрев модели на %s (голос: %s)…",
                    device.upper(), Path(voice).name)
        t0 = time.time()
        gpt, spk = xtts_model.get_conditioning_latents(audio_path=voice)
        with torch.inference_mode():
            _ = xtts_model.inference(
                text="Прогрев, раз, два, три.",
                language=LANGUAGE,
                gpt_cond_latent=gpt,
                speaker_embedding=spk,
                **XTTS_KWARGS,
            )
        logger.info("Прогрев завершён за %.1f с", time.time() - t0)
    except Exception:
        logger.exception("Прогрев не удался (не критично)")


def _voice_folder_signature(voice_dir: Path) -> str:
    """Хэш по всем *.wav внутри папки: имя|size|mtime."""
    parts: list[str] = []
    for f in sorted(voice_dir.glob("*.wav")):
        try:
            st = f.stat()
            parts.append(f"{f.name}|{st.st_size}|{int(st.st_mtime)}")
        except OSError:
            continue
    if not parts:
        return "empty"
    return "||".join(parts)


def _cache_file_for(voice_dir: Path) -> Path:
    sig = _voice_folder_signature(voice_dir)
    h = hashlib.md5(sig.encode("utf-8")).hexdigest()[:16]
    safe = re.sub(r"[^\w\-]+", "_", voice_dir.name, flags=re.UNICODE).strip("_") or "voice"
    return CACHE_DIR / f"{safe}__{h}.pt"


def _list_voice_dirs() -> list[Path]:
    result: list[Path] = []
    for d in sorted(VOICES_DIR.iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if any(d.glob("*.wav")):
            result.append(d)
    return result


def _open_local_path(path: Path) -> None:
    try:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
    except Exception:
        logger.exception("Не удалось открыть папку: %s", path)


# ============================================================ Токенизация
def _sentenize(text: str) -> list[str]:
    """Разбивает текст на предложения, сохраняя пунктуацию.

    Приоритет: razdel → pysbd → regex-фолбэк.
    razdel понимает русские сокращения («т.д.», «г.», «ул.» и т.п.).
    """
    if not text:
        return []

    if _HAS_RAZDEL and _razdel_sentenize is not None:
        try:
            return [s.text.strip() for s in _razdel_sentenize(text) if s.text.strip()]
        except Exception:
            logger.exception("razdel.sentenize упал, откат на pysbd/regex")

    if _HAS_PYSBD and _PY_SBD_SEG is not None:
        try:
            return [s.strip() for s in _PY_SBD_SEG.segment(text) if s.strip()]
        except Exception:
            logger.exception("pysbd упал, откат на regex")

    # Простой фолбэк по знакам конца предложения
    parts = re.split(r"(?<=[.!?…])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _split_long_sentence(sentence: str, max_len: int) -> list[str]:
    """Режет слишком длинное предложение по границам придаточных,
    а если не помогло — по пробелам."""
    # Сначала пытаемся по запятым/точкам с запятой/тире
    parts = re.split(r"(?<=[,;:—])\s+", sentence)
    chunks: list[str] = []
    cur = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_len:
            cur = cur + " " + p
        else:
            chunks.append(cur)
            cur = p
    if cur:
        chunks.append(cur)

    # Кусок всё ещё длинный — рубим по словам
    result: list[str] = []
    for c in chunks:
        if len(c) <= max_len:
            result.append(c)
            continue
        buf = ""
        for w in c.split():
            if len(w) > max_len:
                if buf:
                    result.append(buf)
                    buf = ""
                for k in range(0, len(w), max_len):
                    result.append(w[k:k + max_len])
                continue
            if not buf:
                buf = w
            elif len(buf) + 1 + len(w) <= max_len:
                buf = buf + " " + w
            else:
                result.append(buf)
                buf = w
        if buf:
            result.append(buf)
    return result


def _split_into_chunks(text: str, max_len: int) -> list[str]:
    """Разбивает текст на чанки, группируя короткие предложения.

    Алгоритм:
      1. Нормализуем пробелы.
      2. Разбиваем на предложения через razdel/pysbd/regex.
      3. Сливаем последовательные предложения, пока очередное
         влезает в max_len.
      4. Слишком длинные предложения режем отдельно (_split_long_sentence).
    """
    if not text:
        return []

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    sentences = _sentenize(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for s in sentences:
        # Отдельно длинное предложение — режем сами, не сливаем с соседями
        if len(s) > max_len:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_sentence(s, max_len))
            continue

        if not current:
            current = s
        elif len(current) + 1 + len(s) <= max_len:
            current = current + " " + s
        else:
            chunks.append(current)
            current = s

    if current:
        chunks.append(current)

    return [c for c in chunks if c]


# ============================================================ QThread workers
class ModelLoader(QThread):
    status_changed = pyqtSignal(str)
    loaded = pyqtSignal(object, int, int, int, str)  # tts, sr, cl, cm, device
    failed = pyqtSignal(str)

    def __init__(self, want_gpu: bool) -> None:
        super().__init__()
        self._want_gpu = want_gpu

    def run(self) -> None:
        try:
            _register_xtts_safe_globals()

            logger.info("Загрузка модели %s …", MODEL_NAME)
            from TTS.api import TTS  # type: ignore

            want_gpu = self._want_gpu and _CUDA["available"]
            initial_device = "cuda" if want_gpu else "cpu"
            self.status_changed.emit(
                f"⏳ Загрузка XTTS-v2 на {initial_device.upper()}…"
            )

            tts = TTS(MODEL_NAME)
            tts.to("cpu")

            try:
                synth = getattr(tts, "synthesizer", None)
                model = getattr(synth, "tts_model", None) if synth else None
                if model is not None:
                    model.eval()
            except Exception:
                pass

            synth = getattr(tts, "synthesizer", None)
            sr = getattr(synth, "output_sample_rate", None) if synth else None
            sr = int(sr) if sr else 24000

            char_limit, chunk_max = _detect_char_limits(synth)

            device = "cpu"
            if initial_device == "cuda":
                if _try_move_to_cuda(tts):
                    device = "cuda"
                else:
                    self.status_changed.emit(
                        "⚠ Не удалось использовать GPU, работаем на CPU"
                    )

            self.status_changed.emit("🔥 Прогрев модели…")
            _warmup(tts, device)

            logger.info(
                "Модель готова: устройство=%s, sr=%d Гц, char_limit=%d, chunk_max=%d",
                device.upper(), sr, char_limit, chunk_max,
            )
            self.loaded.emit(tts, sr, char_limit, chunk_max, device)
        except Exception as e:
            logger.exception("Ошибка загрузки модели")
            self.failed.emit(str(e))


class DeviceSwitcher(QThread):
    status_changed = pyqtSignal(str)
    done = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, tts: Any, current_device: str, target: str) -> None:
        super().__init__()
        self._tts = tts
        self._current = current_device
        self._target = target

    def run(self) -> None:
        try:
            logger.info("Переключение устройства: %s → %s",
                        self._current.upper(), self._target.upper())
            t0 = time.time()
            if self._target == "cuda":
                if not _try_move_to_cuda(self._tts):
                    self.failed.emit("CUDA OOM или ошибка — остаёмся на CPU")
                    return
            else:
                _move_to_cpu(self._tts)

            self.status_changed.emit(f"🔥 Прогрев на {self._target.upper()}…")
            _warmup(self._tts, self._target)
            logger.info("Переключение завершено за %.1f с", time.time() - t0)
            self.done.emit(self._target)
        except Exception as e:
            logger.exception("Ошибка переключения устройства")
            self.failed.emit(str(e))


class TrainWorker(QThread):
    """Считает латенты по всем *.wav в папке голоса и сохраняет в кэш."""
    status_changed = pyqtSignal(str)
    finished_ok = pyqtSignal(str, str)
    failed = pyqtSignal(str)

    def __init__(self, tts: Any, voice_dir: Path, device: str) -> None:
        super().__init__()
        self._tts = tts
        self._voice_dir = voice_dir
        self._device = device

    def run(self) -> None:
        try:
            wavs = sorted(self._voice_dir.glob("*.wav"))
            if not wavs:
                self.failed.emit(f"В папке «{self._voice_dir.name}» нет .wav")
                return

            logger.info("Обучение голоса «%s» (%d файлов) на %s",
                        self._voice_dir.name, len(wavs), self._device.upper())
            self.status_changed.emit(
                f"🧬 Обучение «{self._voice_dir.name}» ({len(wavs)} записей)…"
            )
            t0 = time.time()

            synth = getattr(self._tts, "synthesizer", None)
            if synth is None:
                self.failed.emit("Модель не инициализирована")
                return
            xtts_model = synth.tts_model

            paths = [str(p) for p in wavs]
            gpt_cond, spk_emb = xtts_model.get_conditioning_latents(audio_path=paths)

            cache_path = _cache_file_for(self._voice_dir)
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
            logger.info("Обучение «%s» завершено за %.1f с",
                        self._voice_dir.name, dt)
            self.finished_ok.emit(
                self._voice_dir.name,
                f"Голос «{self._voice_dir.name}» обучен на {len(wavs)} записях "
                f"({dt:.1f} с)",
            )
        except Exception as e:
            logger.exception("Ошибка обучения голоса")
            self.failed.emit(str(e))


class SynthWorker(QThread):
    status_changed = pyqtSignal(str)
    progress_changed = pyqtSignal(float)
    elapsed_changed = pyqtSignal(str)
    finished_ok = pyqtSignal(str, float, float)
    failed = pyqtSignal(str)

    def __init__(
        self,
        tts: Any,
        sample_rate: int,
        text: str,
        voice_dir: Path,
        cleanup: bool,
        out_path: Path,
        chunk_max: int,
        char_limit: int,
        device: str,
        latents_cache: dict,
        stop_check,
    ) -> None:
        super().__init__()
        self._tts = tts
        self._sr = sample_rate
        self._text = text
        self._voice_dir = voice_dir
        self._cleanup = cleanup
        self._out_path = out_path
        self._chunk_max = chunk_max
        self._char_limit = char_limit
        self._device = device
        self._latents_cache = latents_cache
        self._stop_check = stop_check

    def _get_latents(self) -> tuple[Any, Any]:
        key = (str(self._voice_dir), self._device)
        cached = self._latents_cache.get(key)
        if cached is not None:
            logger.debug("Латенты из памяти: %s", self._voice_dir.name)
            return cached

        cache_path = _cache_file_for(self._voice_dir)
        if cache_path.exists():
            try:
                logger.info("Латенты из диска: %s", cache_path.name)
                data = torch.load(cache_path, map_location="cpu", weights_only=False)
                gpt = data["gpt_cond"].to(self._device)
                spk = data["spk_emb"].to(self._device)
                pair = (gpt, spk)
                self._latents_cache[key] = pair
                return pair
            except Exception:
                logger.exception("Кэш повреждён, пересчитываем: %s", cache_path)
                try:
                    cache_path.unlink()
                except Exception:
                    pass

        self.status_changed.emit("🧬 Подготовка голоса…")
        wavs = sorted(self._voice_dir.glob("*.wav"))
        if not wavs:
            raise RuntimeError(f"В папке «{self._voice_dir.name}» нет .wav")
        logger.info("Расчёт латентов: %s (%d файлов, на %s)",
                    self._voice_dir.name, len(wavs), self._device.upper())

        synth = getattr(self._tts, "synthesizer", None)
        if synth is None:
            raise RuntimeError("TTS-модель не инициализирована (synthesizer is None)")
        xtts_model = synth.tts_model

        gpt_cond, spk_emb = xtts_model.get_conditioning_latents(
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

        gpt_cond = gpt_cond.to(self._device)
        spk_emb = spk_emb.to(self._device)
        self._latents_cache[key] = (gpt_cond, spk_emb)
        return gpt_cond, spk_emb

    @staticmethod
    def _hard_trim(text: str, limit: int) -> str:
        if len(text) <= limit:
            return text
        cut = text[:limit]
        idx = cut.rfind(" ")
        if idx < limit * 0.5:
            return cut
        return cut[:idx].rstrip()

    def _synthesize_raw(self, text: str, gpt_cond: Any, spk_emb: Any) -> np.ndarray:
        """Чистый инференс без постобработки. Постобработку делаем один раз
        на всём аудио — иначе громкость прыгает между чанками."""
        synth = getattr(self._tts, "synthesizer", None)
        if synth is None:
            raise RuntimeError("TTS-модель не инициализирована")
        xtts_model = synth.tts_model

        if len(text) > self._char_limit:
            trimmed = self._hard_trim(text, self._chunk_max)
            logger.warning("Обрезан слишком длинный фрагмент (%d → %d симв.)",
                           len(text), len(trimmed))
            text = trimmed

        with torch.inference_mode():
            out = xtts_model.inference(
                text=text,
                language=LANGUAGE,
                gpt_cond_latent=gpt_cond,
                speaker_embedding=spk_emb,
                **XTTS_KWARGS,
            )

        audio = np.asarray(out["wav"], dtype=np.float32).reshape(-1)
        return audio

    def run(self) -> None:
        t_start = time.time()
        try:
            gpt_cond, spk_emb = self._get_latents()

            chunks = _split_into_chunks(self._text, max_len=self._chunk_max)
            total = max(1, len(chunks))
            if not chunks:
                self.failed.emit("Не удалось разбить текст на чанки")
                return

            logger.info("Разбито на %d фрагмент(ов) по ≤ %d симв.",
                        total, self._chunk_max)
            for i, c in enumerate(chunks, 1):
                logger.debug("  [%d/%d] %d симв.: %s%s",
                             i, total, len(c), c[:60],
                             "…" if len(c) > 60 else "")

            parts: list[np.ndarray] = []
            for i, chunk in enumerate(chunks, 1):
                if self._stop_check():
                    logger.info("Прервано пользователем на %d/%d", i, total)
                    break

                t_chunk = time.time()
                self.status_changed.emit(f"🎧 Синтез… {i}/{total}")
                logger.info("[%d/%d] (%d симв.)", i, total, len(chunk))

                audio = self._synthesize_raw(chunk, gpt_cond, spk_emb)
                parts.append(audio)

                dt_chunk = time.time() - t_chunk
                pct = (i / total) * 100.0
                self.progress_changed.emit(pct)

                elapsed_total = time.time() - t_start
                avg = elapsed_total / i
                remaining = avg * (total - i)
                eta_min = int(remaining // 60)
                eta_sec = int(remaining % 60)
                eta_str = f"{eta_min}м {eta_sec:02d}с" if eta_min else f"{eta_sec}с"
                logger.info("[%d/%d] готово за %.1f с (осталось ~%s)",
                            i, total, dt_chunk, eta_str)
                self.elapsed_changed.emit(
                    f"⏱ {int(elapsed_total // 60)}м {int(elapsed_total % 60):02d}с, "
                    f"ETA ~{eta_str}"
                )

            if not parts:
                self.failed.emit("Ничего не сгенерировано")
                return

            # Склейка с паузами
            if len(parts) == 1:
                audio = parts[0]
            else:
                pause = np.zeros(int(self._sr * INSERT_PAUSE_MS / 1000.0),
                                 dtype=np.float32)
                pieces: list[np.ndarray] = []
                for i, p in enumerate(parts):
                    pieces.append(p)
                    if i < len(parts) - 1:
                        pieces.append(pause)
                audio = np.concatenate(pieces)

            audio = audio.astype(np.float32)

            # Постобработка один раз на всём аудио
            if self._cleanup:
                audio = _postprocess(audio, self._sr)
            else:
                peak = float(np.max(np.abs(audio))) or 1.0
                audio = audio / peak * 0.95

            self.status_changed.emit("💾 Сохранение файла…")
            ext = self._out_path.suffix.lower()
            if ext == ".mp3":
                _write_mp3(str(self._out_path), audio, self._sr)
            else:
                _write_wav(str(self._out_path), audio, self._sr)

            dur = len(audio) / self._sr
            elapsed = time.time() - t_start
            rtf = dur / elapsed if elapsed > 0 else 0.0
            logger.info(
                "Сохранено: %s (%.1f с аудио за %.1f с синтеза, ×%.2f realtime, %s)",
                self._out_path, dur, elapsed, rtf, self._device.upper(),
            )
            self.finished_ok.emit(self._out_path.name, dur, elapsed)
        except Exception as e:
            logger.exception("Ошибка синтеза")
            self.failed.emit(str(e))


class PreviewWorker(QThread):
    status_changed = pyqtSignal(str)
    finished_ok = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(
        self,
        tts: Any,
        sample_rate: int,
        text: str,
        voice_dir: Path,
        cleanup: bool,
        char_limit: int,
        chunk_max: int,
        device: str,
        latents_cache: dict,
    ) -> None:
        super().__init__()
        self._tts = tts
        self._sr = sample_rate
        self._text = text
        self._voice_dir = voice_dir
        self._cleanup = cleanup
        self._char_limit = char_limit
        self._chunk_max = chunk_max
        self._device = device
        self._latents_cache = latents_cache

    def run(self) -> None:
        try:
            self.status_changed.emit("🔊 Проверка голоса…")
            wavs = sorted(self._voice_dir.glob("*.wav"))
            if not wavs:
                self.failed.emit(f"В папке «{self._voice_dir.name}» нет .wav")
                return

            synth = getattr(self._tts, "synthesizer", None)
            if synth is None:
                self.failed.emit("Модель не инициализирована")
                return
            xtts_model = synth.tts_model

            key = (str(self._voice_dir), self._device)
            lat = self._latents_cache.get(key)
            if lat is None:
                cache_path = _cache_file_for(self._voice_dir)
                if cache_path.exists():
                    try:
                        data = torch.load(cache_path, map_location="cpu",
                                          weights_only=False)
                        gpt = data["gpt_cond"].to(self._device)
                        spk = data["spk_emb"].to(self._device)
                    except Exception:
                        logger.exception("Кэш повреждён, пересчитаем")
                        gpt, spk = xtts_model.get_conditioning_latents(
                            audio_path=[str(p) for p in wavs]
                        )
                else:
                    logger.info("Латентов нет — считаем для превью")
                    gpt, spk = xtts_model.get_conditioning_latents(
                        audio_path=[str(p) for p in wavs]
                    )
                self._latents_cache[key] = (gpt, spk)
            else:
                gpt, spk = lat

            text = self._text
            if len(text) > self._char_limit:
                text = text[: self._chunk_max].rsplit(" ", 1)[0]

            with torch.inference_mode():
                out = xtts_model.inference(
                    text=text,
                    language=LANGUAGE,
                    gpt_cond_latent=gpt,
                    speaker_embedding=spk,
                    **XTTS_KWARGS,
                )
            audio = np.asarray(out["wav"], dtype=np.float32).reshape(-1)
            if self._cleanup:
                audio = _postprocess(audio, self._sr)
            else:
                peak = float(np.max(np.abs(audio))) or 1.0
                audio = audio / peak * 0.95

            sd.play(audio, self._sr)
            sd.wait()
            self.finished_ok.emit()
        except Exception as e:
            logger.exception("Ошибка проверки голоса")
            self.failed.emit(str(e))


# ============================================================ Постобработка
def _postprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    if audio.size == 0:
        return audio

    a = audio.astype(np.float32, copy=True)
    a -= float(np.mean(a))

    if sps is not None:
        nyq = sr / 2.0
        hp = min(HP_FREQ, nyq * 0.9) / nyq
        try:
            sos = sps.butter(4, hp, btype="highpass", output="sos")
            a = sps.sosfiltfilt(sos, a).astype(np.float32)
        except Exception:
            logger.exception("High-pass filter failed")

        lp = min(LP_FREQ, nyq * 0.95) / nyq
        try:
            sos = sps.butter(6, lp, btype="lowpass", output="sos")
            a = sps.sosfiltfilt(sos, a).astype(np.float32)
        except Exception:
            logger.exception("Low-pass filter failed")

    try:
        win = max(1, int(sr * 0.02))
        kernel = np.ones(win, dtype=np.float32) / win
        env = np.convolve(np.abs(a), kernel, mode="same")
        thr = 10.0 ** (GATE_THRESH_DB / 20.0)
        gain = np.where(env < thr, np.maximum(env / thr, 0.15), 1.0)
        a *= gain.astype(np.float32)
    except Exception:
        logger.exception("Noise gate failed")

    fade = max(1, int(sr * FADE_MS / 1000.0))
    if a.size > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        a[:fade] *= ramp
        a[-fade:] *= ramp[::-1]

    peak = float(np.max(np.abs(a))) or 1.0
    a = a / peak * 0.95
    return a.astype(np.float32)


def _write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def _write_mp3(path: str, audio: np.ndarray, sr: int, bitrate: int = MP3_BITRATE) -> None:
    try:
        import lameenc  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "Для сохранения MP3 нужен пакет lameenc. Установите: pip install lameenc"
        ) from e

    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16).tobytes()
    enc = lameenc.Encoder()
    enc.set_bit_rate(bitrate)
    enc.set_in_sample_rate(sr)
    enc.set_channels(1)
    enc.set_quality(2)
    data = enc.encode(pcm) + enc.flush()
    with open(path, "wb") as f:
        f.write(data)


# ============================================================ MainWindow
class TTSApp(QMainWindow):
    sig_status = pyqtSignal(str)
    sig_progress = pyqtSignal(float)
    sig_elapsed = pyqtSignal(str)
    sig_enable = pyqtSignal(bool)

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Читалка — XTTS-v2 (PyQt6, мультизапись голоса)")
        self.resize(1080, 860)
        self.setMinimumSize(860, 620)

        self.tts: Optional[Any] = None
        self.sample_rate: int = 24000
        self.device: str = "cpu"
        self.char_limit: int = DEFAULT_CHAR_LIMIT
        self.chunk_max: int = max(MIN_CHUNK_LEN,
                                  DEFAULT_CHAR_LIMIT - CHUNK_SAFETY_MARGIN)

        self._latents_cache: dict[tuple[str, str], tuple[Any, Any]] = {}
        self.is_working: bool = False
        self.stop_flag: bool = False
        self.accentizer: Optional[Any] = None

        self._loader_thread: Optional[ModelLoader] = None
        self._switch_thread: Optional[DeviceSwitcher] = None
        self._worker_thread: Optional[QThread] = None

        self.sig_status.connect(self._on_status)
        self.sig_progress.connect(self._on_progress)
        self.sig_elapsed.connect(self._on_elapsed)
        self.sig_enable.connect(self._on_enable)

        self._build_ui()
        self._refresh_voice_list()
        self._init_accentizer()
        self._load_model_async()

        logger.info("Приложение запущено.")
        logger.info("Логических ядер CPU: %d", _CPU_COUNT)
        if _CUDA["available"]:
            logger.info("GPU: %s, VRAM: %.1f ГБ (порог FP16: %.1f ГБ)",
                        _CUDA["name"], _CUDA["vram_gb"], FP16_VRAM_THRESHOLD_GB)
        else:
            logger.info("GPU: недоступна, работаем на CPU")

        if _HAS_RAZDEL:
            logger.info("Токенизация: razdel")
        elif _HAS_PYSBD:
            logger.info("Токенизация: pysbd (razdel не установлен)")
        else:
            logger.info("Токенизация: regex (razdel/pysbd не установлены)")

        logger.info("Голоса (папки): %s", VOICES_DIR)
        logger.info("Кэш латентов:   %s", CACHE_DIR)
        logger.info("Вывод:          %s", OUTPUT_DIR)

    # --------------------------------------------------------------- UI
    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(10, 10, 10, 8)
        root_layout.setSpacing(6)

        top = QHBoxLayout()
        top.setSpacing(6)

        self.btn_open = QPushButton("📂 Открыть .txt")
        self.btn_synth = QPushButton("🎧 Озвучить в файл")
        self.btn_train = QPushButton("🎓 Обучить голос")
        self.btn_stop = QPushButton("⏹ Стоп")
        self.btn_clear = QPushButton("🗑 Очистить")
        self.btn_open_out = QPushButton("🗂 Открыть output")

        for b in (self.btn_open, self.btn_synth, self.btn_train,
                  self.btn_stop, self.btn_clear):
            top.addWidget(b)
        top.addStretch(1)
        top.addWidget(self.btn_open_out)

        self.btn_open.clicked.connect(self.open_file)
        self.btn_synth.clicked.connect(self.synthesize_to_file)
        self.btn_train.clicked.connect(self.train_voice)
        self.btn_stop.clicked.connect(self.stop)
        self.btn_clear.clicked.connect(self._clear_text)
        self.btn_open_out.clicked.connect(lambda: _open_local_path(OUTPUT_DIR))

        root_layout.addLayout(top)

        opts = QHBoxLayout()
        opts.setSpacing(6)

        opts.addWidget(QLabel("Голос:"))
        self.voice_combo = QComboBox()
        self.voice_combo.setMinimumWidth(220)
        opts.addWidget(self.voice_combo)

        self.btn_add_voice = QPushButton("➕ Голос")
        self.btn_open_voices = QPushButton("📁 Голоса")
        self.btn_preview = QPushButton("🔊 Проверка")
        opts.addWidget(self.btn_add_voice)
        opts.addWidget(self.btn_open_voices)
        opts.addWidget(self.btn_preview)

        opts.addSpacing(10)
        opts.addWidget(QLabel("Формат:"))
        self.format_combo = QComboBox()
        self.format_combo.addItems(["MP3", "WAV"])
        self.format_combo.setCurrentText("MP3")
        self.format_combo.setFixedWidth(80)
        opts.addWidget(self.format_combo)

        opts.addSpacing(10)
        self.chk_cleanup = QCheckBox("🧹 Очистка")
        self.chk_cleanup.setChecked(True)
        opts.addWidget(self.chk_cleanup)

        self.chk_stress = QCheckBox("Ё/ударения")
        self.chk_stress.setChecked(_HAS_RUACCENT)
        self.chk_stress.setEnabled(_HAS_RUACCENT)
        opts.addWidget(self.chk_stress)

        self.chk_gpu = QCheckBox("🎮 GPU")
        self.chk_gpu.setChecked(_default_use_gpu())
        self.chk_gpu.setEnabled(_CUDA["available"])
        opts.addWidget(self.chk_gpu)

        gpu_label_text = (
            f"{_CUDA['name']} · {_CUDA['vram_gb']:.0f} ГБ"
            if _CUDA["available"] else "CUDA не обнаружена"
        )
        self.lbl_gpu = QLabel(gpu_label_text)
        self.lbl_gpu.setStyleSheet("color:#888;")
        opts.addWidget(self.lbl_gpu)

        opts.addStretch(1)
        root_layout.addLayout(opts)

        self.btn_add_voice.clicked.connect(self.add_voice)
        self.btn_open_voices.clicked.connect(lambda: _open_local_path(VOICES_DIR))
        self.btn_preview.clicked.connect(self.preview_voice)
        self.chk_gpu.toggled.connect(self._on_gpu_toggle)

        self.text_area = QPlainTextEdit()
        f = QFont("Segoe UI", 12)
        self.text_area.setFont(f)
        self.text_area.setPlainText(
            "Как пользоваться:\n\n"
            "1. Каждый голос — отдельная папка в D:\\Reader\\voices\\.\n"
            "   Внутри папки — один или несколько .wav (6–15 с каждый, без шума).\n"
            "   Чем больше чистых записей — тем стабильнее клон.\n"
            "2. Кнопка «➕ Голос» — создать новую папку голоса из выбранных .wav.\n"
            "3. Кнопка «🎓 Обучить голос» — один раз считает латенты по всем .wav\n"
            "   в выбранной папке и кэширует их (voices/.cache).\n"
            "4. «🎧 Озвучить в файл» — синтез текущего текста.\n\n"
            "Текст автоматически режется на фразы по точкам/запятым\n"
            "(через razdel — корректно для русского), короткие предложения\n"
            "склеиваются в один чанк, между чанками вставляются паузы 150 мс.\n\n"
            "Галочка «Ё/ударения» — автоударения через ruaccent.\n"
            "Галочка «🎮 GPU» — CUDA-ускорение, FP16 при VRAM < 3 ГБ.\n"
            "Файлы сохраняются в output/."
        )
        root_layout.addWidget(self.text_area, stretch=1)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        root_layout.addWidget(line)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        root_layout.addWidget(self.progress)

        info = QHBoxLayout()
        self.lbl_status = QLabel("⏳ Загрузка модели…")
        self.lbl_elapsed = QLabel("")
        self.lbl_elapsed.setStyleSheet("color:#888;")
        info.addWidget(self.lbl_status)
        info.addStretch(1)
        info.addWidget(self.lbl_elapsed)
        root_layout.addLayout(info)

        hint = QLabel(
            "Лицензия XTTS-v2 (CPML) допускает только некоммерческое использование."
        )
        hint.setStyleSheet("color:#888;")
        root_layout.addWidget(hint)

        self._action_buttons = [
            self.btn_open, self.btn_synth, self.btn_train,
            self.btn_clear, self.btn_open_out, self.btn_add_voice,
            self.btn_open_voices, self.btn_preview,
        ]
        self.btn_stop.setEnabled(False)

    # ------------------------------------------------------ voice folders
    def _refresh_voice_list(self) -> None:
        current = self.voice_combo.currentText()
        dirs = _list_voice_dirs()
        self.voice_combo.clear()
        for d in dirs:
            self.voice_combo.addItem(d.name)
        if current and self.voice_combo.findText(current) >= 0:
            self.voice_combo.setCurrentText(current)
        logger.debug("Список голосов: %s",
                     [d.name for d in dirs] or "— пусто —")

    def _current_voice_dir(self) -> Optional[Path]:
        name = self.voice_combo.currentText().strip()
        if not name:
            return None
        d = VOICES_DIR / name
        if d.is_dir() and any(d.glob("*.wav")):
            return d
        return None

    def add_voice(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Выберите .wav-записи одного голоса (можно несколько)",
            str(BASE_DIR),
            "Аудио (*.wav);;Все файлы (*.*)",
        )
        if not files:
            return

        name, ok = QInputDialog.getText(
            self, "Имя голоса", "Название папки голоса:"
        )
        if not ok or not name.strip():
            return

        safe = re.sub(r"[^\w\-]+", "_", name.strip(), flags=re.UNICODE).strip("_")
        if not safe:
            QMessageBox.warning(self, "Ошибка", "Недопустимое имя папки.")
            return

        dst_dir = VOICES_DIR / safe
        dst_dir.mkdir(parents=True, exist_ok=True)

        added = 0
        for src in files:
            try:
                src_p = Path(src)
                if src_p.suffix.lower() == ".wav":
                    shutil.copyfile(src_p, dst_dir / src_p.name)
                else:
                    import soundfile as sf  # type: ignore
                    data, sr = sf.read(src, always_2d=False)
                    if data.ndim > 1:
                        data = data.mean(axis=1)
                    out_name = src_p.with_suffix(".wav").name
                    sf.write(dst_dir / out_name, data.astype(np.float32), sr)
                added += 1
            except Exception:
                logger.exception("Не удалось добавить %s", src)

        logger.info("Голос «%s»: добавлено %d файл(ов) → %s",
                    safe, added, dst_dir)
        self._refresh_voice_list()
        idx = self.voice_combo.findText(safe)
        if idx >= 0:
            self.voice_combo.setCurrentIndex(idx)
        self.sig_status.emit(f"➕ Голос «{safe}»: {added} файл(ов)")

    # ------------------------------------------------------ accentizer
    def _init_accentizer(self) -> None:
        if not _HAS_RUACCENT:
            logger.info("ruaccent не установлен — автоударения недоступны "
                        "(pip install ruaccent)")
            return
        if RUAccent is None:
            logger.warning("ruaccent помечен доступным, но класс недоступен")
            return
        try:
            acc = RUAccent()
            acc.load(omograph_model_size="turbo", use_dictionary=True)
            self.accentizer = acc
            logger.info("ruaccent загружен — автоударения доступны")
        except Exception:
            logger.exception("Не удалось инициализировать ruaccent")
            self.accentizer = None

    def _apply_stress(self, text: str) -> str:
        if self.accentizer is None:
            return text
        try:
            t0 = time.time()
            processed = self.accentizer.process_all(text)
            dt = time.time() - t0
            added = processed.count(STRESS_MARK) - text.count(STRESS_MARK)
            words = len(re.findall(r"\w+", text))
            logger.info("Ударения: +%d меток на %d слов (%.1f с)",
                        added, words, dt)
            return processed
        except Exception:
            logger.exception("Ошибка расстановки ударений — беру исходный текст")
            return text

    # ------------------------------------------------------ model loading
    def _load_model_async(self) -> None:
        want_gpu = self.chk_gpu.isChecked()
        self._loader_thread = ModelLoader(want_gpu)
        self._loader_thread.status_changed.connect(self.sig_status.emit)
        self._loader_thread.loaded.connect(self._on_model_loaded)
        self._loader_thread.failed.connect(self._on_model_failed)
        self._loader_thread.start()

    def _on_model_loaded(
        self, tts: Any, sr: int, char_limit: int, chunk_max: int, device: str
    ) -> None:
        self.tts = tts
        self.sample_rate = sr
        self.char_limit = char_limit
        self.chunk_max = chunk_max
        self.device = device
        self.chk_gpu.setChecked(device == "cuda")
        self.sig_status.emit(f"✅ XTTS-v2 готов ({device.upper()}, {sr} Гц)")
        self.sig_enable.emit(True)

    def _on_model_failed(self, err: str) -> None:
        self.sig_status.emit(f"❌ Ошибка загрузки модели: {err}")
        QMessageBox.critical(self, "Ошибка загрузки модели", err)

    # ------------------------------------------------------ GPU toggle
    def _on_gpu_toggle(self, checked: bool) -> None:
        if self.is_working:
            logger.warning("Переключение устройства невозможно: идёт синтез")
            self.chk_gpu.blockSignals(True)
            self.chk_gpu.setChecked(self.device == "cuda")
            self.chk_gpu.blockSignals(False)
            return
        target = "cuda" if (checked and _CUDA["available"]) else "cpu"
        if target == self.device or self.tts is None:
            return

        self.is_working = True
        self.sig_enable.emit(False)
        self.sig_status.emit(f"🔄 Переключение на {target.upper()}…")

        self._switch_thread = DeviceSwitcher(self.tts, self.device, target)
        self._switch_thread.status_changed.connect(self.sig_status.emit)
        self._switch_thread.done.connect(self._on_device_switched)
        self._switch_thread.failed.connect(self._on_device_switch_failed)
        self._switch_thread.finished.connect(self._on_worker_done)
        self._switch_thread.start()

    def _on_device_switched(self, device: str) -> None:
        self.device = device
        self._latents_cache.clear()
        self.chk_gpu.blockSignals(True)
        self.chk_gpu.setChecked(device == "cuda")
        self.chk_gpu.blockSignals(False)
        self.sig_status.emit(f"✅ Готово ({device.upper()})")

    def _on_device_switch_failed(self, err: str) -> None:
        logger.warning("Переключение устройства: %s", err)
        self.sig_status.emit(f"⚠ {err}")
        self.chk_gpu.blockSignals(True)
        self.chk_gpu.setChecked(False)
        self.chk_gpu.blockSignals(False)

    # ------------------------------------------------------ train voice
    def train_voice(self) -> None:
        if self.tts is None:
            self.sig_status.emit("⚠ Модель ещё загружается")
            return
        if self.is_working:
            return
        voice_dir = self._current_voice_dir()
        if voice_dir is None:
            self.sig_status.emit("⚠ Выберите голос")
            return

        old_cache = _cache_file_for(voice_dir)
        if old_cache.exists():
            try:
                old_cache.unlink()
                logger.info("Старый кэш удалён: %s", old_cache.name)
            except Exception:
                logger.exception("Не удалось удалить старый кэш")

        for k in list(self._latents_cache):
            if k[0] == str(voice_dir):
                del self._latents_cache[k]

        self.is_working = True
        self.stop_flag = False
        self.sig_enable.emit(False)
        self.sig_progress.emit(0.0)
        self.sig_status.emit("🎓 Обучение голоса…")

        self._worker_thread = TrainWorker(self.tts, voice_dir, self.device)
        self._worker_thread.status_changed.connect(self.sig_status.emit)
        self._worker_thread.finished_ok.connect(self._on_train_done)
        self._worker_thread.failed.connect(self._on_worker_error)
        self._worker_thread.finished.connect(self._on_worker_done)
        self._worker_thread.start()

    def _on_train_done(self, voice_name: str, msg: str) -> None:
        self.sig_progress.emit(100.0)
        self.sig_status.emit(f"✅ {msg}")
        QMessageBox.information(self, "Обучение завершено", msg)

    # ------------------------------------------------------ synthesis
    def synthesize_to_file(self) -> None:
        if self.tts is None:
            self.sig_status.emit("⚠ Модель ещё загружается")
            return
        if self.is_working:
            return

        voice_dir = self._current_voice_dir()
        if voice_dir is None:
            self.sig_status.emit("⚠ Выберите голос")
            return

        text = self.text_area.toPlainText().strip()
        if not text:
            logger.info("Пустой текст")
            return

        if self.chk_stress.isChecked() and self.accentizer is not None:
            self.sig_status.emit("📖 Расстановка ударений…")
            text = self._apply_stress(text)

        fmt = self.format_combo.currentText().upper()
        ext = ".mp3" if fmt == "MP3" else ".wav"
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"output_{ts}{ext}"

        cleanup = self.chk_cleanup.isChecked()

        self.is_working = True
        self.stop_flag = False
        self.sig_enable.emit(False)
        self.sig_progress.emit(0.0)
        self.sig_status.emit("🎧 Синтез…")
        self.sig_elapsed.emit("")

        logger.info(
            "Синтез: %d симв., голос=%s, формат=%s, очистка=%s, ударения=%s, "
            "устройство=%s → %s (чанк ≤ %d)",
            len(text), voice_dir.name, fmt, cleanup,
            bool(self.chk_stress.isChecked() and self.accentizer is not None),
            self.device.upper(), out_path.name, self.chunk_max,
        )

        self._worker_thread = SynthWorker(
            tts=self.tts,
            sample_rate=self.sample_rate,
            text=text,
            voice_dir=voice_dir,
            cleanup=cleanup,
            out_path=out_path,
            chunk_max=self.chunk_max,
            char_limit=self.char_limit,
            device=self.device,
            latents_cache=self._latents_cache,
            stop_check=lambda: self.stop_flag,
        )
        self._worker_thread.status_changed.connect(self.sig_status.emit)
        self._worker_thread.progress_changed.connect(self.sig_progress.emit)
        self._worker_thread.elapsed_changed.connect(self.sig_elapsed.emit)
        self._worker_thread.finished_ok.connect(self._on_synth_done)
        self._worker_thread.failed.connect(self._on_worker_error)
        self._worker_thread.finished.connect(self._on_worker_done)
        self._worker_thread.start()

    def _on_synth_done(self, out_name: str, dur: float, elapsed: float) -> None:
        rtf = dur / elapsed if elapsed > 0 else 0.0
        self.sig_progress.emit(100.0)
        self.sig_status.emit(f"✅ Готово: {out_name} ({dur:.0f} с)")
        self.sig_elapsed.emit(f"⏱ {elapsed:.0f} с, ×{rtf:.1f}")

    # ------------------------------------------------------ preview
    def preview_voice(self) -> None:
        if self.tts is None or self.is_working:
            return
        voice_dir = self._current_voice_dir()
        if voice_dir is None:
            self.sig_status.emit("⚠ Выберите голос")
            return

        text = "Привет! Так звучит этот голос. Раз, два, три, четыре, пять."
        if self.chk_stress.isChecked() and self.accentizer is not None:
            text = self._apply_stress(text)

        self.is_working = True
        self.stop_flag = False
        self.sig_enable.emit(False)
        self.sig_progress.emit(0.0)
        self.sig_status.emit("🔊 Проверка голоса…")
        logger.info("Проверка голоса: %s (%s)", voice_dir.name, self.device.upper())

        self._worker_thread = PreviewWorker(
            tts=self.tts,
            sample_rate=self.sample_rate,
            text=text,
            voice_dir=voice_dir,
            cleanup=self.chk_cleanup.isChecked(),
            char_limit=self.char_limit,
            chunk_max=self.chunk_max,
            device=self.device,
            latents_cache=self._latents_cache,
        )
        self._worker_thread.status_changed.connect(self.sig_status.emit)
        self._worker_thread.finished_ok.connect(
            lambda: self.sig_status.emit("✅ Проверка завершена")
        )
        self._worker_thread.failed.connect(self._on_worker_error)
        self._worker_thread.finished.connect(self._on_worker_done)
        self._worker_thread.start()

    # ------------------------------------------------------ worker lifecycle
    def _on_worker_done(self) -> None:
        self.is_working = False
        self.sig_enable.emit(True)

    def _on_worker_error(self, err: str) -> None:
        logger.error("Worker error: %s", err)
        self.sig_status.emit(f"❌ {err}")
        if self.tts is not None and self.device == "cuda" \
                and "out of memory" in err.lower():
            try:
                _move_to_cpu(self.tts)
            except Exception:
                logger.exception("Не удалось перейти на CPU")
            self.device = "cpu"
            self._latents_cache.clear()
            self.chk_gpu.blockSignals(True)
            self.chk_gpu.setChecked(False)
            self.chk_gpu.blockSignals(False)
            self.sig_status.emit("⚠ OOM: откатились на CPU, попробуйте ещё раз")

    # ------------------------------------------------------ Stop
    def stop(self) -> None:
        if not self.is_working:
            return
        logger.info("Запрошена остановка")
        self.stop_flag = True
        try:
            sd.stop()
        except Exception:
            pass
        self.sig_status.emit("⏹ Остановка…")

    # ------------------------------------------------------ File / utils
    def open_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть текст", str(BASE_DIR),
            "Текстовые файлы (*.txt);;Все файлы (*.*)",
        )
        if not path:
            return
        content = ""
        for enc in ("utf-8", "cp1251", "koi8-r"):
            try:
                with open(path, "r", encoding=enc) as f:
                    content = f.read()
                logger.info("Открыт файл: %s (%s, %d симв.)",
                            path, enc, len(content))
                break
            except UnicodeDecodeError:
                continue
        if not content:
            logger.error("Не удалось определить кодировку: %s", path)
            QMessageBox.warning(self, "Ошибка", "Не удалось прочитать файл.")
            return
        self.text_area.setPlainText(content)
        self.sig_status.emit(f"📄 {Path(path).name}")

    def _clear_text(self) -> None:
        self.text_area.clear()

    # ------------------------------------------------------ signal slots
    def _on_status(self, msg: str) -> None:
        self.lbl_status.setText(msg)

    def _on_progress(self, value: float) -> None:
        v = int(max(0.0, min(100.0, float(value))))
        self.progress.setValue(v)

    def _on_elapsed(self, text: str) -> None:
        self.lbl_elapsed.setText(text)

    def _on_enable(self, enabled: bool) -> None:
        for b in self._action_buttons:
            b.setEnabled(enabled)
        self.btn_stop.setEnabled(not enabled and self.is_working)
        self.chk_gpu.setEnabled(enabled and _CUDA["available"])

    # ------------------------------------------------------ close
    def closeEvent(self, a0: Optional[QCloseEvent]) -> None:  # noqa: N802
        self.stop_flag = True
        for th in (self._worker_thread, self._switch_thread, self._loader_thread):
            if th is not None and th.isRunning():
                th.wait(2000)
        if a0 is not None:
            a0.accept()


# ============================================================ entry point
def main() -> None:
    if sys.platform == "win32":
        try:
            from ctypes import windll  # type: ignore
            windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("XTTS Reader")
    win = TTSApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()