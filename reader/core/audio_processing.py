"""Постобработка аудио: фильтры, noise-gate, фейды, нормализация и запись файлов."""
from __future__ import annotations

import logging
import wave

import numpy as np

from reader.core import config

logger = logging.getLogger("reader.audio")

try:
    from scipy import signal as sps
except Exception:
    sps = None  # type: ignore


def postprocess(audio: np.ndarray, sr: int) -> np.ndarray:
    """Убирает постоянную составляющую, шум, щелчки; нормализует громкость."""
    if audio.size == 0:
        return audio

    a = audio.astype(np.float32, copy=True)
    a -= float(np.mean(a))

    if sps is not None:
        nyq = sr / 2.0
        hp = min(config.HP_FREQ, nyq * 0.9) / nyq
        try:
            sos = sps.butter(4, hp, btype="highpass", output="sos")
            a = sps.sosfiltfilt(sos, a).astype(np.float32)
        except Exception:
            logger.exception("High-pass filter failed")

        lp = min(config.LP_FREQ, nyq * 0.95) / nyq
        try:
            sos = sps.butter(6, lp, btype="lowpass", output="sos")
            a = sps.sosfiltfilt(sos, a).astype(np.float32)
        except Exception:
            logger.exception("Low-pass filter failed")

    try:
        win = max(1, int(sr * 0.02))
        kernel = np.ones(win, dtype=np.float32) / win
        env = np.convolve(np.abs(a), kernel, mode="same")
        thr = 10.0 ** (config.GATE_THRESH_DB / 20.0)
        gain = np.where(env < thr, np.maximum(env / thr, 0.15), 1.0)
        a *= gain.astype(np.float32)
    except Exception:
        logger.exception("Noise gate failed")

    fade = max(1, int(sr * config.FADE_MS / 1000.0))
    if a.size > 2 * fade:
        ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
        a[:fade] *= ramp
        a[-fade:] *= ramp[::-1]

    peak = float(np.max(np.abs(a))) or 1.0
    a = a / peak * 0.95
    return a.astype(np.float32)


def normalize_peak(audio: np.ndarray, target: float = 0.95) -> np.ndarray:
    """Простая нормализация по пику — без фильтрации."""
    peak = float(np.max(np.abs(audio))) or 1.0
    return (audio / peak * target).astype(np.float32)


def write_wav(path: str, audio: np.ndarray, sr: int) -> None:
    pcm = np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def write_mp3(path: str, audio: np.ndarray, sr: int,
              bitrate: int = config.MP3_BITRATE) -> None:
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
