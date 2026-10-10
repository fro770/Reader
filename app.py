import sys
import os
import re
import json
import time
import shutil
import logging
import wave
import threading
from pathlib import Path
from typing import Optional, Any, Callable, Dict, List

# === Фиксация пути к espeak-ng для русского TTS ===
_ESPEAK_DIR = r"C:\Program Files\espeak-ng"
if os.path.isdir(_ESPEAK_DIR):
    _current_path = os.environ.get("PATH", "")
    if _ESPEAK_DIR not in _current_path:
        os.environ["PATH"] = _ESPEAK_DIR + os.pathsep + _current_path
    os.environ["PHONEMIZER_ESPEAK_PATH"] = os.path.join(_ESPEAK_DIR, "espeak-ng.exe")
    os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = os.path.join(_ESPEAK_DIR, "libespeak-ng.dll")
# =================================================

# Настройка потоков CPU до импорта torch
_CPU_COUNT = os.cpu_count() or 4
for env_var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ.setdefault(env_var, str(_CPU_COUNT))

import numpy as np
import torch

# ==============================================================================
# ПАТЧ 1: СОВМЕСТИМОСТЬ С НОВЫМ PYTORCH (2.6+)
# ==============================================================================
_original_torch_load = torch.load

def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)

torch.load = _patched_torch_load

# ==============================================================================
# ПАТЧ 2: ЧТЕНИЕ АУДИО ЧЕРЕЗ SOUNDFILE (обход torchcodec/FFmpeg)
# ==============================================================================
import torchaudio
import soundfile as sf

def _patched_torchaudio_load(filepath, *args, **kwargs):
    assert sf is not None
    data, sr = sf.read(str(filepath), dtype='float32', always_2d=True)
    if data.shape[1] > 1:
        data = data.mean(axis=1)
    else:
        data = data[:, 0]
    waveform = torch.from_numpy(data).unsqueeze(0)
    return waveform, int(sr)

torchaudio.load = _patched_torchaudio_load
# ==============================================================================

# --- Опциональные зависимости ---
try:
    import sounddevice as sd
except ImportError:
    sd = None  # type: ignore

try:
    from scipy import signal as sps
    HAS_SCIPY = True
except ImportError:
    sps = None  # type: ignore
    HAS_SCIPY = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    librosa = None  # type: ignore
    HAS_LIBROSA = False

try:
    HAS_SOUNDFILE = True
except ImportError:
    sf = None  # type: ignore
    HAS_SOUNDFILE = False

try:
    from ruaccent import RUAccent
    HAS_RUACCENT = True
except ImportError:
    RUAccent = None  # type: ignore
    HAS_RUACCENT = False

try:
    import lameenc
    HAS_LAME = True
except ImportError:
    lameenc = None  # type: ignore
    HAS_LAME = False

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QComboBox, QCheckBox, QProgressBar,
    QTextEdit, QFileDialog, QMessageBox, QGroupBox,
    QTabWidget, QListWidget, QDialog,
    QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QObject, QThread, pyqtSignal
from PyQt6.QtGui import QFont

# ------------------------------------------------------------------ Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reader")

# ------------------------------------------------------------------ Config
BASE_DIR = Path(__file__).parent
VOICES_DIR = BASE_DIR / "voices"
OUTPUT_DIR = BASE_DIR / "output"
TEXTS_DIR = BASE_DIR / "texts"

for d in (VOICES_DIR, OUTPUT_DIR, TEXTS_DIR):
    d.mkdir(exist_ok=True)

LANGUAGE = "ru"
MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
STRESS_MARK = "\u0301"

# ВАЖНО: enable_text_splitting=False — используем свой сплиттер
XTTS_KWARGS = dict(
    temperature=0.75,
    length_penalty=1.0,
    repetition_penalty=5.0,
    top_k=50,
    top_p=0.85,
    enable_text_splitting=False,
)

DEFAULT_OMOGRAPHS: Dict[str, List[str]] = {
    "замок":    [f"за{STRESS_MARK}мок", f"замо{STRESS_MARK}к"],
    "мука":     [f"му{STRESS_MARK}ка", f"мука{STRESS_MARK}"],
    "муки":     [f"му{STRESS_MARK}ки", f"муки{STRESS_MARK}"],
    "муку":     [f"му{STRESS_MARK}ку", f"муку{STRESS_MARK}"],
    "мои":      [f"мои{STRESS_MARK}", f"мо{STRESS_MARK}и"],
    "мира":     [f"ми{STRESS_MARK}ра", f"мира{STRESS_MARK}"],
    "дороги":   [f"доро{STRESS_MARK}ги", f"дороги{STRESS_MARK}"],
    "сторона":  [f"сто{STRESS_MARK}рона", f"сторона{STRESS_MARK}"],
    "языки":    [f"язы{STRESS_MARK}ки", f"языки{STRESS_MARK}"],
    "руки":     [f"ру{STRESS_MARK}ки", f"руки{STRESS_MARK}"],
    "вино":     [f"вино{STRESS_MARK}", f"ви{STRESS_MARK}но"],
    "плачу":    [f"пла{STRESS_MARK}чу", f"плачу{STRESS_MARK}"],
    "свое":     [f"сво{STRESS_MARK}е", f"свое{STRESS_MARK}"],
    "своё":     [f"сво{STRESS_MARK}ё", f"свое{STRESS_MARK}"],
    "пою":      [f"пою{STRESS_MARK}", f"по{STRESS_MARK}ю"],
    "около":    [f"о{STRESS_MARK}коло", f"около{STRESS_MARK}"],
    "наркотики":[f"наркоти{STRESS_MARK}ки", f"нарко{STRESS_MARK}тики"],
    "кафе":     [f"кафе{STRESS_MARK}", f"ка{STRESS_MARK}фе"],
}

DARK_STYLE = """
    QMainWindow { background-color: #2b2b2b; color: #eeeeee; }
    QTabWidget::pane { border: 1px solid #444; background-color: #2b2b2b; }
    QTabBar::tab { background-color: #333; color: #ccc; padding: 8px 16px;
                   border: 1px solid #444; border-bottom: none;
                   border-top-left-radius: 4px; border-top-right-radius: 4px;
                   margin-right: 2px; }
    QTabBar::tab:selected { background-color: #2b2b2b; color: #fff;
                            border-bottom: 2px solid #0078d7; }
    QPushButton { background-color: #0078d7; color: white; border: none;
                  padding: 6px 12px; border-radius: 4px; font-weight: bold; }
    QPushButton:hover { background-color: #005a9e; }
    QPushButton:disabled { background-color: #555; color: #888; }
    QPushButton#danger { background-color: #d9534f; }
    QPushButton#danger:hover { background-color: #c9302c; }
    QTextEdit, QListWidget { background-color: #1e1e1e; color: #eee;
                             border: 1px solid #444; border-radius: 4px;
                             font-size: 14px; }
    QComboBox { background-color: #333; color: #eee; border: 1px solid #444;
                padding: 4px; border-radius: 4px; min-width: 150px; }
    QComboBox QAbstractItemView { background-color: #333; color: #eee;
                                  selection-background-color: #0078d7; }
    QGroupBox { border: 1px solid #444; border-radius: 4px; margin-top: 10px;
                font-weight: bold; }
    QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
    QProgressBar { border: 1px solid #444; border-radius: 4px; text-align: center;
                   background-color: #1e1e1e; color: white; height: 20px; }
    QProgressBar::chunk { background-color: #0078d7; border-radius: 3px; }
    QListWidget::item:selected { background-color: #0078d7; }
    QLabel { color: #eee; }
    QCheckBox { color: #eee; spacing: 5px; }
    QRadioButton { color: #eee; font-size: 14px; padding: 4px; }
"""

# ------------------------------------------------------------------ Signals
class WorkerSignals(QObject):
    progress = pyqtSignal(int)
    eta = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    ask_omograph = pyqtSignal(str, str, list, int, object)


# ------------------------------------------------------------------ XTTS Manager
class XTTSManager:
    def __init__(self) -> None:
        self.tts: Any = None
        self.xtts_model: Any = None
        self.device: str = "cpu"
        self.sample_rate: int = 24000

    def load_model(self, device: str = "cpu") -> None:
        from TTS.api import TTS  # type: ignore
        logger.info("Загрузка модели %s на %s...", MODEL_NAME, device)
        self.tts = TTS(MODEL_NAME)
        self.tts.to(device)
        self.device = device

        synth = getattr(self.tts, "synthesizer", None)
        if synth is not None and hasattr(synth, "tts_model"):
            self.xtts_model = synth.tts_model
        elif hasattr(self.tts, "tts_model"):
            self.xtts_model = self.tts.tts_model
        else:
            self.xtts_model = self.tts

        if synth is not None and hasattr(synth, "output_sample_rate"):
            self.sample_rate = int(synth.output_sample_rate)

    def get_latents(self, audio_path: str):
        return self.xtts_model.get_conditioning_latents(audio_path=audio_path)

    def synthesize(self, text: str, gpt_cond: Any, spk_emb: Any) -> np.ndarray:
        # Очищаем текст от символов ударения (espeak-ng их не понимает)
        clean_text = text.replace(STRESS_MARK, "")
        
        logger.debug("Синтез текста: %s", clean_text[:100])
        
        out = self.xtts_model.inference(
            text=clean_text,
            language=LANGUAGE,
            gpt_cond_latent=gpt_cond,
            speaker_embedding=spk_emb,
            **XTTS_KWARGS,
        )
        return np.asarray(out["wav"], dtype=np.float32).reshape(-1)

    def get_latents_from_long_audio(self, audio_path: str, temp_dir: Path):
        if not (HAS_LIBROSA and HAS_SOUNDFILE):
            logger.warning("librosa/soundfile недоступны — использую файл целиком")
            return self.get_latents(audio_path)

        assert librosa is not None and sf is not None

        logger.info("Анализ длинного аудио: %s", Path(audio_path).name)
        y, sr_raw = librosa.load(audio_path, sr=None, mono=True)
        sr = int(sr_raw)
        duration = len(y) / sr

        if duration <= 12.0:
            return self.get_latents(audio_path)

        intervals = librosa.effects.split(y, top_db=25)
        merged: List[tuple] = []
        current: Optional[tuple] = None
        max_sec, target_sec = 12.0, 10.0

        for start, end in intervals:
            if current is None:
                current = (start, end)
            elif (end - current[0]) / sr <= max_sec:
                current = (current[0], end)
            else:
                merged.append(current)
                current = (start, end)
        if current is not None:
            merged.append(current)

        if not merged:
            step = int(target_sec * sr)
            merged = [(i * step, min((i + 1) * step, len(y)))
                      for i in range(len(y) // step + 1)]

        final_chunks: List[tuple] = []
        for start, end in merged:
            if (end - start) / sr <= max_sec:
                final_chunks.append((start, end))
            else:
                step = int(target_sec * sr)
                for i in range(start, end, step):
                    e = min(i + step, end)
                    if (e - i) / sr >= 3.0:
                        final_chunks.append((i, e))

        temp_dir.mkdir(parents=True, exist_ok=True)
        chunk_files: List[Path] = []
        for i, (s, e) in enumerate(final_chunks):
            chunk_path = temp_dir / f"chunk_{i:03d}.wav"
            sf.write(chunk_path, y[s:e], sr)
            chunk_files.append(chunk_path)

        spk_embs = []
        best_gpt = None
        for i, cf in enumerate(chunk_files):
            logger.info("Извлечение профиля %d/%d", i + 1, len(chunk_files))
            try:
                gpt, spk = self.get_latents(str(cf))
                spk_embs.append(spk.cpu())
                if best_gpt is None:
                    best_gpt = gpt.cpu()
            except Exception as e:
                logger.warning("Пропуск куска %d: %s", i + 1, e)

        for f in chunk_files:
            try:
                f.unlink()
            except OSError:
                pass

        if not spk_embs:
            raise ValueError("Не удалось извлечь ни одного валидного куска из аудио.")

        avg_spk = torch.stack(spk_embs).mean(dim=0)
        return best_gpt, avg_spk


# ------------------------------------------------------------------ Workers
class ModelLoaderWorker(QObject):
    signals = WorkerSignals()

    def __init__(self, xtts: XTTSManager, use_gpu: bool) -> None:
        super().__init__()
        self.xtts = xtts
        self.use_gpu = use_gpu

    def run(self) -> None:
        try:
            device = "cuda" if self.use_gpu and torch.cuda.is_available() else "cpu"
            self.signals.eta.emit("Загрузка XTTS-v2...")
            logger.info("Загрузка модели на %s...", device.upper())
            self.xtts.load_model(device)
            self.signals.eta.emit("Инициализация завершена")
            self.signals.finished.emit(True, f"Модель готова ({device.upper()})")
        except Exception as e:
            logger.exception("Ошибка загрузки модели")
            self.signals.finished.emit(False, str(e))


class TrainWorker(QObject):
    signals = WorkerSignals()

    def __init__(self, xtts: XTTSManager, voice_dir: Path) -> None:
        super().__init__()
        self.xtts = xtts
        self.voice_dir = voice_dir

    def run(self) -> None:
        try:
            wavs = sorted(self.voice_dir.glob("*.wav"))
            if not wavs:
                raise ValueError("В папке нет .wav файлов")

            spk_embs: List[Any] = []
            best_gpt: Any = None
            total = len(wavs)
            temp_dir = self.voice_dir / ".cache" / "temp_chunks"

            t_start = time.time()
            for i, wav in enumerate(wavs):
                elapsed = time.time() - t_start
                if i > 0:
                    avg = elapsed / i
                    remaining = avg * (total - i)
                    eta_str = f"{int(remaining // 60)}м {int(remaining % 60):02d}с"
                else:
                    eta_str = "расчёт..."
                self.signals.eta.emit(f"Обучение {i+1}/{total} | ETA: {eta_str}")
                
                logger.info("Анализ %d/%d: %s", i + 1, total, wav.name)
                self.signals.progress.emit(int((i / total) * 100))
                
                try:
                    gpt, spk = self.xtts.get_latents_from_long_audio(str(wav), temp_dir)
                    spk_embs.append(spk if not hasattr(spk, "cpu") else spk.cpu())
                    if best_gpt is None:
                        assert gpt is not None
                        best_gpt = gpt.cpu() if hasattr(gpt, "cpu") else gpt
                except Exception as e:
                    logger.warning("Пропуск файла %s: %s", wav.name, e)

            if best_gpt is None or not spk_embs:
                raise RuntimeError("Не удалось извлечь профиль ни из одного файла")

            avg_spk = torch.stack(spk_embs).mean(dim=0)
            cache_dir = self.voice_dir / ".cache"
            cache_dir.mkdir(exist_ok=True)
            torch.save({"gpt_cond": best_gpt, "spk_emb": avg_spk},
                       cache_dir / "profile.pt")

            with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump({"wavs": [w.name for w in wavs],
                           "trained_at": time.time()}, f)

            if temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)

            self.signals.progress.emit(100)
            self.signals.finished.emit(True, f"Профиль обучен на {len(spk_embs)} файлах")
        except Exception as e:
            logger.exception("Ошибка обучения")
            self.signals.finished.emit(False, str(e))


class AccentWorker(QObject):
    signals = WorkerSignals()

    def __init__(self, accentizer: Any, text: str,
                 omograph_dict: Dict[str, List[str]]) -> None:
        super().__init__()
        self.accentizer = accentizer
        self.text = text
        self.omograph_dict = omograph_dict
        self.results: Dict[int, int] = {}

    def run(self) -> None:
        try:
            sentences = re.split(r'(?<=[.!?…])\s+', self.text)
            omograph_id = 0

            for sent_idx, sent in enumerate(sentences):
                words = re.findall(r'[а-яёА-ЯЁ]+', sent)
                for word in words:
                    lower = word.lower()
                    options = self.omograph_dict.get(lower)
                    if not options or len(options) < 2:
                        continue

                    self.signals.eta.emit(f"Ожидание выбора ударения для: {word}")
                    event = threading.Event()
                    self.signals.ask_omograph.emit(sent, word, options,
                                                   omograph_id, event)
                    event.wait()

                    selected_idx = self.results.get(omograph_id, 0)
                    selected_word = options[selected_idx]

                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    new_sent = pattern.sub(selected_word, sent, count=1)
                    self.text = self.text.replace(sent, new_sent, 1)
                    sent = new_sent
                    omograph_id += 1

            if self.accentizer is not None:
                self.signals.eta.emit("Финальная обработка текста...")
                logger.info("Финальная расстановка ударений...")
                try:
                    self.text = self.accentizer.process_all(
                        self.text, put_yo=True, put_stress_homo=False
                    )
                except TypeError:
                    self.text = self.accentizer.process_all(self.text)

            self.signals.finished.emit(True, self.text)
        except Exception as e:
            logger.exception("Ошибка акцентуатора")
            self.signals.finished.emit(False, str(e))


class SynthWorker(QObject):
    signals = WorkerSignals()

    def __init__(self, xtts: XTTSManager, voice_dir: Path, text: str,
                 out_dir: Path, settings: dict) -> None:
        super().__init__()
        self.xtts = xtts
        self.voice_dir = voice_dir
        self.text = text
        self.out_dir = out_dir
        self.settings = settings
        self.is_running = True

    def stop(self) -> None:
        self.is_running = False

    def is_uptodate(self) -> bool:
        cache_dir = self.voice_dir / ".cache"
        meta_file = cache_dir / "meta.json"
        profile_file = cache_dir / "profile.pt"
        if not profile_file.exists() or not meta_file.exists():
            return False
        try:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            current_wavs = sorted(w.name for w in self.voice_dir.glob("*.wav"))
            return current_wavs == sorted(meta.get("wavs", []))
        except Exception:
            return False

    def train_profile(self) -> None:
        wavs = sorted(self.voice_dir.glob("*.wav"))
        if not wavs:
            raise ValueError("Нет файлов для обучения")

        spk_embs: List[Any] = []
        best_gpt: Any = None
        temp_dir = self.voice_dir / ".cache" / "temp_chunks"

        for i, wav in enumerate(wavs):
            logger.info("Переобучение %d/%d", i + 1, len(wavs))
            try:
                gpt, spk = self.xtts.get_latents_from_long_audio(str(wav), temp_dir)
                spk_embs.append(spk if not hasattr(spk, "cpu") else spk.cpu())
                if best_gpt is None:
                    assert gpt is not None
                    best_gpt = gpt.cpu() if hasattr(gpt, "cpu") else gpt
            except Exception as e:
                logger.warning("Пропуск файла %s: %s", wav.name, e)

        if best_gpt is None or not spk_embs:
            raise RuntimeError("Не удалось извлечь профиль ни из одного файла")

        avg_spk = torch.stack(spk_embs).mean(dim=0)
        cache_dir = self.voice_dir / ".cache"
        cache_dir.mkdir(exist_ok=True)
        torch.save({"gpt_cond": best_gpt, "spk_emb": avg_spk},
                   cache_dir / "profile.pt")
        with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump({"wavs": [w.name for w in wavs],
                       "trained_at": time.time()}, f)

        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def normalize_text(text: str) -> str:
        """Превращает числа и символы в слова для правильного чтения espeak-ng."""
        # Убираем символ ударения
        text = text.replace(STRESS_MARK, "")
        
        # Годы 1900-2099 → прописью
        def year_repl(m):
            y = m.group(0)
            try:
                n = int(y)
                if 2000 <= n <= 2099:
                    rest = n - 2000
                    if rest == 0:
                        return "две тысячи"
                    elif rest < 20:
                        units = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
                                 "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
                                 "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
                        return f"две тысячи {units[rest]}"
                    else:
                        units = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
                        tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
                        t = tens[rest // 10]
                        u = units[rest % 10]
                        return f"две тысячи {t} {u}".strip()
                elif 1900 <= n <= 1999:
                    rest = n - 1900
                    if rest < 20:
                        units = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
                                 "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
                                 "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
                        return f"тысяча девятьсот {units[rest]}"
                    else:
                        units = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
                        tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
                        t = tens[rest // 10]
                        u = units[rest % 10]
                        return f"тысяча девятьсот {t} {u}".strip()
            except:
                pass
            return y
        
        # Числа до 999 → прописью
        def num_repl(m):
            n = m.group(0)
            try:
                num = int(n)
                units = ["", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять",
                         "десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать",
                         "шестнадцать", "семнадцать", "восемнадцать", "девятнадцать"]
                tens = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
                hundreds = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
                
                if num == 0:
                    return "ноль"
                if num < 0 or num > 999:
                    return n
                
                result = []
                if num >= 100:
                    result.append(hundreds[num // 100])
                    num %= 100
                if num >= 20:
                    result.append(tens[num // 10])
                    num %= 10
                if num > 0:
                    result.append(units[num])
                return " ".join(result)
            except:
                return n
        
        # Сначала годы, потом остальные числа
        text = re.sub(r'\b(19|20)\d{2}\b', year_repl, text)
        text = re.sub(r'\b\d+\b', num_repl, text)
        
        # Символы → слова
        text = text.replace('%', ' процентов ')
        text = text.replace('$', ' долларов ')
        text = text.replace('€', ' евро ')
        text = text.replace('№', ' номер ')
        text = text.replace('&', ' и ')
        text = text.replace('@', ' собака ')
        text = text.replace('#', ' номер ')
        text = text.replace('+', ' плюс ')
        text = text.replace('=', ' равно ')
        
        # Множественные пробелы
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    @staticmethod
    def split_text(text: str, max_len: int = 180) -> List[str]:
        """Делим текст на куски по предложениям, не длиннее max_len символов."""
        text = re.sub(r"\s+", " ", text).strip()
        sentences = re.split(r"(?<=[.!?…])\s+", text)
        chunks: List[str] = []
        for s in sentences:
            if len(s) <= max_len:
                chunks.append(s)
                continue
            current = ""
            for w in s.split():
                if len(current) + len(w) + 1 <= max_len:
                    current = (current + " " + w).strip()
                else:
                    if current:
                        chunks.append(current)
                    current = w
            if current:
                chunks.append(current)
        return chunks

    def postprocess(self, audio: np.ndarray) -> np.ndarray:
        if not HAS_SCIPY or sps is None:
            return audio
        a = audio.astype(np.float32)
        a -= float(np.mean(a))
        nyq = self.xtts.sample_rate / 2.0
        sos = sps.butter(4, min(70.0, nyq * 0.9) / nyq,
                         btype="highpass", output="sos")
        a = sps.sosfiltfilt(sos, a).astype(np.float32)
        peak = float(np.max(np.abs(a))) or 1.0
        return (a / peak * 0.95).astype(np.float32)

    def save_audio(self, audio: np.ndarray, path: Path) -> None:
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
        if path.suffix.lower() == ".mp3" and HAS_LAME and lameenc is not None:
            enc = lameenc.Encoder()
            enc.set_bit_rate(192)
            enc.set_in_sample_rate(int(self.xtts.sample_rate))
            enc.set_channels(1)
            enc.set_quality(2)
            data = enc.encode(pcm.tobytes()) + enc.flush()
            with open(path, "wb") as f:
                f.write(data)
        else:
            with wave.open(str(path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(int(self.xtts.sample_rate))
                wf.writeframes(pcm.tobytes())

    def run(self) -> None:
        try:
            if not self.is_uptodate():
                logger.info("Профиль устарел. Переобучение...")
                self.signals.eta.emit("Переобучение профиля...")
                self.train_profile()

            logger.info("Загрузка профиля...")
            self.signals.eta.emit("Загрузка профиля...")
            cache_file = self.voice_dir / ".cache" / "profile.pt"
            data: Dict[str, Any] = torch.load(cache_file, map_location=self.xtts.device)
            
            gpt_cond: Any = data["gpt_cond"]
            spk_emb: Any = data["spk_emb"]
            
            gpt_cond = gpt_cond.to(self.xtts.device)
            spk_emb = spk_emb.to(self.xtts.device)

            # Делим текст на куски
            chunks = self.split_text(self.text)
            # Нормализуем каждый кусок (числа → слова, символы → слова)
            chunks = [self.normalize_text(c) for c in chunks]
            # Убираем пустые куски
            chunks = [c for c in chunks if c.strip()]
            
            total = max(1, len(chunks))
            audios: List[np.ndarray] = []

            t_start = time.time()
            for i, chunk in enumerate(chunks):
                if not self.is_running:
                    break
                elapsed = time.time() - t_start
                if i > 0:
                    avg = elapsed / i
                    remaining = avg * (total - i)
                    eta_str = f"{int(remaining // 60)}м {int(remaining % 60):02d}с"
                else:
                    eta_str = "расчёт..."
                self.signals.eta.emit(f"Синтез {i+1}/{total} | ETA: {eta_str}")
                
                logger.info("Синтез %d/%d: %s...", i + 1, total, chunk[:50])
                audio = self.xtts.synthesize(chunk, gpt_cond, spk_emb)
                if self.settings.get("cleanup"):
                    audio = self.postprocess(audio)
                audios.append(audio)
                
                # Пауза 150мс между кусками для естественности
                pause_samples = int(self.xtts.sample_rate * 0.15)
                audios.append(np.zeros(pause_samples, dtype=np.float32))
                
                self.signals.progress.emit(int((i + 1) / total * 100))

            if not audios:
                raise RuntimeError("Не удалось сгенерировать аудио")

            final_audio = np.concatenate(audios)
            ts = time.strftime("%Y%m%d_%H%M%S")
            fmt = self.settings.get("format", "WAV")
            ext = ".mp3" if fmt == "MP3" else ".wav"
            out_path = self.out_dir / f"output_{ts}{ext}"

            logger.info("Сохранение файла...")
            self.signals.eta.emit("Сохранение файла...")
            self.save_audio(final_audio, out_path)
            self.signals.finished.emit(True, str(out_path))

        except Exception as e:
            logger.exception("Ошибка синтеза")
            self.signals.finished.emit(False, str(e))


# ------------------------------------------------------------------ Omograph Dialog
class OmographDialog(QDialog):
    def __init__(self, sentence: str, word: str, options: List[str],
                 parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Спорное ударение (омограф)")
        self.setWindowFlags(self.windowFlags() |
                            Qt.WindowType.WindowStaysOnTopHint)
        self.resize(520, 300)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Найдено слово с неоднозначным ударением:"))

        pattern = re.compile(re.escape(word), re.IGNORECASE)
        highlighted = pattern.sub(
            f"<b style='color:#4aa3ff;font-size:18px;'>{word}</b>", sentence
        )
        lbl_context = QLabel(highlighted)
        lbl_context.setWordWrap(True)
        lbl_context.setTextFormat(Qt.TextFormat.RichText)
        lbl_context.setStyleSheet(
            "font-size:16px;padding:10px;background-color:#1e1e1e;"
            "border-radius:4px;"
        )
        layout.addWidget(lbl_context)
        layout.addWidget(QLabel("Выберите правильное произношение:"))

        self.btn_group = QButtonGroup(self)
        for i, opt in enumerate(options):
            rb = QRadioButton(self.stress_to_capital(opt))
            if i == 0:
                rb.setChecked(True)
            self.btn_group.addButton(rb, i)
            layout.addWidget(rb)

        btn_ok = QPushButton("Подтвердить выбор")
        btn_ok.setDefault(True)
        btn_ok.clicked.connect(self.accept)
        layout.addWidget(btn_ok)

        QApplication.beep()

    @staticmethod
    def stress_to_capital(text: str) -> str:
        result: List[str] = []
        i = 0
        while i < len(text):
            ch = text[i]
            if i + 1 < len(text) and text[i + 1] == STRESS_MARK:
                result.append(ch.upper())
                i += 2
            else:
                result.append(ch)
                i += 1
        return "".join(result)

    def get_selected(self) -> int:
        return self.btn_group.checkedId()


# ------------------------------------------------------------------ Main Window
class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Russian TTS Reader (XTTS-v2 Pro)")
        self.resize(1000, 700)

        self.xtts = XTTSManager()
        self.is_model_loaded = False

        self.model_thread: Optional[QThread] = None
        self.model_worker: Optional[ModelLoaderWorker] = None

        self.current_thread: Optional[QThread] = None
        self.current_worker: Optional[QObject] = None

        self.accent_thread: Optional[QThread] = None
        self.accent_worker: Optional[AccentWorker] = None

        self._pending_synth_voice: str = ""
        self._pending_synth_settings: dict = {}

        self.accentizer: Any = None
        self.omograph_dict: Dict[str, List[str]] = dict(DEFAULT_OMOGRAPHS)

        if HAS_RUACCENT and RUAccent is not None:
            try:
                self.accentizer = RUAccent()
                self.accentizer.load(omograph_model_size="turbo",
                                     use_dictionary=True)
                inner = getattr(self.accentizer, "omograph_dict", None)
                if isinstance(inner, dict) and inner:
                    self.omograph_dict.update(inner)
                logger.info("RUAccent загружен. Омографов в словаре: %d",
                            len(self.omograph_dict))
            except Exception:
                logger.exception("Не удалось загрузить RUAccent")
                self.accentizer = None

        self._indicator_color = "yellow"
        self._indicator_text = "Загрузка модели..."
        self._prev_color = "yellow"
        self._prev_text = "Загрузка модели..."

        self.setup_ui()
        self.refresh_voices()
        self.start_model_loading()

    def setup_ui(self) -> None:
        self.setStyleSheet(DARK_STYLE)
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        # Индикатор состояния в правом верхнем углу
        top_bar = QHBoxLayout()
        top_bar.addStretch()
        self.indicator = QLabel("● Загрузка модели...")
        self.indicator.setStyleSheet("color: #f0ad4e; font-weight: bold; padding: 5px; font-size: 14px;")
        top_bar.addWidget(self.indicator)
        main_layout.addLayout(top_bar)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)

        self.tab_synth = QWidget()
        self.tab_voices = QWidget()
        self.tab_settings = QWidget()

        self.tabs.addTab(self.tab_synth, "🎙 Синтез")
        self.tabs.addTab(self.tab_voices, "🧠 Голоса и Обучение")
        self.tabs.addTab(self.tab_settings, "⚙ Настройки")

        self.build_synth_tab()
        self.build_voices_tab()
        self.build_settings_tab()

    def set_indicator(self, color: str, text: str):
        colors = {
            "green": "#5cb85c",
            "yellow": "#f0ad4e",
            "red": "#d9534f",
            "blue": "#5bc0de",
        }
        css_color = colors.get(color, "#5cb85c")
        self.indicator.setStyleSheet(f"color: {css_color}; font-weight: bold; padding: 5px; font-size: 14px;")
        self.indicator.setText(f"● {text}")
        self._indicator_color = color
        self._indicator_text = text

    def build_synth_tab(self) -> None:
        layout = QVBoxLayout(self.tab_synth)

        top = QHBoxLayout()
        top.addWidget(QLabel("Голос:"))
        self.combo_voices = QComboBox()
        top.addWidget(self.combo_voices)
        top.addSpacing(20)
        top.addWidget(QLabel("Формат:"))
        self.combo_format = QComboBox()
        self.combo_format.addItems(["WAV", "MP3"])
        top.addWidget(self.combo_format)
        top.addStretch()
        layout.addLayout(top)

        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Вставьте текст для озвучивания...")
        layout.addWidget(self.text_edit)

        bottom = QHBoxLayout()
        self.btn_open_txt = QPushButton("📂 Открыть .txt")
        self.btn_open_txt.clicked.connect(self.open_txt)
        self.btn_synth = QPushButton("🚀 Озвучить")
        self.btn_synth.setEnabled(False)
        self.btn_synth.clicked.connect(self.start_synthesis)
        self.btn_stop = QPushButton("⏹ Стоп")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_worker)

        bottom.addWidget(self.btn_open_txt)
        bottom.addStretch()
        bottom.addWidget(self.btn_synth)
        bottom.addWidget(self.btn_stop)
        layout.addLayout(bottom)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        layout.addWidget(self.progress_bar)

    def build_voices_tab(self) -> None:
        layout = QHBoxLayout(self.tab_voices)

        left = QVBoxLayout()
        left.addWidget(QLabel("Список голосов (папки в /voices):"))
        self.voice_list = QListWidget()
        left.addWidget(self.voice_list)
        layout.addLayout(left, 2)

        right = QVBoxLayout()
        self.btn_add = QPushButton("📥 Добавить .wav")
        self.btn_add.clicked.connect(self.add_wavs)
        self.btn_train = QPushButton("🧠 Обучить профиль")
        self.btn_train.clicked.connect(self.start_training)
        self.btn_open_dir = QPushButton("📁 Открыть папку")
        self.btn_open_dir.clicked.connect(self.open_voice_dir)
        self.btn_delete = QPushButton("🗑 Удалить")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(self.delete_voice)

        right.addWidget(self.btn_add)
        right.addWidget(self.btn_train)
        right.addWidget(self.btn_open_dir)
        right.addStretch()
        right.addWidget(self.btn_delete)
        layout.addLayout(right, 1)

    def build_settings_tab(self) -> None:
        layout = QVBoxLayout(self.tab_settings)

        grp_dev = QGroupBox("Устройство")
        lay_dev = QVBoxLayout(grp_dev)
        self.chk_gpu = QCheckBox("Использовать GPU (CUDA)")
        self.chk_gpu.setChecked(torch.cuda.is_available())
        lay_dev.addWidget(self.chk_gpu)
        lay_dev.addWidget(QLabel(f"CUDA доступна: {torch.cuda.is_available()}"))
        layout.addWidget(grp_dev)

        grp_proc = QGroupBox("Обработка")
        lay_proc = QVBoxLayout(grp_proc)
        self.chk_stress = QCheckBox("Интерактивные ударения и Ё (ruaccent)")
        self.chk_stress.setEnabled(HAS_RUACCENT)
        self.chk_stress.setChecked(HAS_RUACCENT)
        self.chk_cleanup = QCheckBox("Очистка шума (High-pass filter)")
        self.chk_cleanup.setChecked(True)
        lay_proc.addWidget(self.chk_stress)
        lay_proc.addWidget(self.chk_cleanup)
        layout.addWidget(grp_proc)

        layout.addStretch()

    def refresh_voices(self) -> None:
        voices = [d.name for d in VOICES_DIR.iterdir()
                  if d.is_dir() and not d.name.startswith(".")]
        self.combo_voices.clear()
        self.combo_voices.addItems(voices)
        self.voice_list.clear()
        self.voice_list.addItems(voices)

    def add_wavs(self) -> None:
        if not self.voice_list.selectedItems():
            return
        voice = self.voice_list.selectedItems()[0].text()
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите .wav", "", "WAV (*.wav)"
        )
        for f in files:
            shutil.copy2(f, VOICES_DIR / voice / Path(f).name)
        logger.info("Добавлено %d файлов в %s", len(files), voice)

    def open_voice_dir(self) -> None:
        if not self.voice_list.selectedItems():
            return
        voice = self.voice_list.selectedItems()[0].text()
        if os.name == "nt":
            os.startfile(VOICES_DIR / voice)

    def delete_voice(self) -> None:
        if not self.voice_list.selectedItems():
            return
        voice = self.voice_list.selectedItems()[0].text()
        ans = QMessageBox.question(
            self, "Удаление", f"Удалить {voice}?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            shutil.rmtree(VOICES_DIR / voice)
            self.refresh_voices()

    def start_model_loading(self) -> None:
        self.model_thread = QThread()
        self.model_worker = ModelLoaderWorker(self.xtts,
                                              self.chk_gpu.isChecked())
        self.model_worker.moveToThread(self.model_thread)

        self.model_thread.started.connect(self.model_worker.run)
        self.model_worker.signals.eta.connect(self.update_eta)
        self.model_worker.signals.finished.connect(self.on_model_loaded)
        self.model_worker.signals.finished.connect(self.model_thread.quit)
        self.model_thread.start()

    def on_model_loaded(self, success: bool, msg: str) -> None:
        if success:
            self.is_model_loaded = True
            self.set_indicator("green", msg)
            self.btn_synth.setEnabled(True)
            logger.info("Модель готова: %s", msg)
        else:
            self.set_indicator("red", f"Ошибка: {msg}")
            logger.error("Не удалось загрузить модель: %s", msg)
            QMessageBox.critical(self, "Ошибка",
                                 f"Не удалось загрузить модель:\n{msg}")

    def start_training(self) -> None:
        if not self.voice_list.selectedItems():
            return
        if not self.is_model_loaded:
            QMessageBox.warning(self, "Ошибка", "Модель ещё не загружена")
            return
        voice = self.voice_list.selectedItems()[0].text()
        self.run_worker(TrainWorker(self.xtts, VOICES_DIR / voice))

    def start_synthesis(self) -> None:
        if not self.is_model_loaded:
            return
        voice = self.combo_voices.currentText()
        if not voice:
            QMessageBox.warning(self, "Ошибка", "Выберите голос")
            return

        text = self.text_edit.toPlainText().strip()
        if not text:
            return

        settings = {
            "format": self.combo_format.currentText(),
            "stress": self.chk_stress.isChecked(),
            "cleanup": self.chk_cleanup.isChecked(),
        }

        if settings["stress"] and self.accentizer is not None:
            logger.info("Анализ текста на омографы...")
            self.accent_thread = QThread()
            self.accent_worker = AccentWorker(self.accentizer, text,
                                              self.omograph_dict)
            self.accent_worker.moveToThread(self.accent_thread)

            self.accent_thread.started.connect(self.accent_worker.run)
            self.accent_worker.signals.ask_omograph.connect(
                self.handle_omograph_request
            )
            self.accent_worker.signals.eta.connect(self.update_eta)
            self.accent_worker.signals.finished.connect(self.on_accent_finished)
            self.accent_worker.signals.finished.connect(self.accent_thread.quit)

            self._pending_synth_voice = voice
            self._pending_synth_settings = settings

            self.btn_synth.setEnabled(False)
            self.set_indicator("blue", "Анализ текста...")
            self.accent_thread.start()
        else:
            self.run_synth_worker(text, voice, settings)

    def handle_omograph_request(self, sentence: str, word: str,
                                options: list, omograph_id: int,
                                event: threading.Event) -> None:
        dlg = OmographDialog(sentence, word, list(options), self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            idx = dlg.get_selected()
        else:
            idx = 0
        if self.accent_worker is not None:
            self.accent_worker.results[omograph_id] = idx
        event.set()

    def on_accent_finished(self, success: bool, result_text: str) -> None:
        if success:
            self.run_synth_worker(result_text,
                                  self._pending_synth_voice,
                                  self._pending_synth_settings)
        else:
            logger.error("Ошибка акцентуатора: %s", result_text)
            self.set_indicator("red", "Ошибка акцентуатора")
            self.btn_synth.setEnabled(True)

    def run_synth_worker(self, text: str, voice: str,
                         settings: dict) -> None:
        self.run_worker(SynthWorker(self.xtts, VOICES_DIR / voice,
                                    text, OUTPUT_DIR, settings))

    def run_worker(self, worker: QObject) -> None:
        if self.current_thread is not None and self.current_thread.isRunning():
            QMessageBox.information(self, "Внимание", "Идёт другой процесс")
            return

        self._prev_color = self._indicator_color
        self._prev_text = self._indicator_text

        self.current_thread = QThread()
        self.current_worker = worker
        worker.moveToThread(self.current_thread)

        signals: WorkerSignals = getattr(worker, "signals")

        self.current_thread.started.connect(getattr(worker, "run"))
        signals.progress.connect(self.progress_bar.setValue)
        signals.eta.connect(self.update_eta)
        signals.finished.connect(self.worker_finished)
        signals.finished.connect(self.current_thread.quit)

        self.btn_synth.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)
        self.set_indicator("blue", "Подготовка...")
        self.current_thread.start()

    def update_eta(self, text: str):
        self.set_indicator("blue", text)

    def worker_finished(self, success: bool, msg: str) -> None:
        self.btn_synth.setEnabled(self.is_model_loaded)
        self.btn_train.setEnabled(self.is_model_loaded)
        self.btn_stop.setEnabled(False)
        if success:
            logger.info("Готово: %s", msg)
            self.set_indicator(self._prev_color, self._prev_text)
            if "output_" in msg:
                QMessageBox.information(self, "Готово",
                                        f"Файл сохранён:\n{msg}")
        else:
            logger.error("Ошибка: %s", msg)
            self.set_indicator("red", f"Ошибка: {msg}")
            QMessageBox.critical(self, "Ошибка", msg)

    def stop_worker(self) -> None:
        if self.current_worker is not None and hasattr(self.current_worker, "stop"):
            getattr(self.current_worker, "stop")()
            logger.info("Остановка процесса...")

    def open_txt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Открыть текст", str(TEXTS_DIR), "Text (*.txt)"
        )
        if path:
            for enc in ("utf-8", "cp1251", "koi8-r"):
                try:
                    with open(path, "r", encoding=enc) as f:
                        self.text_edit.setPlainText(f.read())
                    break
                except UnicodeDecodeError:
                    continue


def main() -> int:
    app = QApplication(sys.argv)
    app.setFont(QFont("Segoe UI", 10))
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())