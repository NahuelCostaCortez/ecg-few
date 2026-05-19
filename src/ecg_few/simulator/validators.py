"""Validator logic for generated beats.

Changing thresholds here changes the meaning of the labels in the dataset.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from .constants import REFERENCE_DURATION_MS, REPOLARIZATION_PIVOT_MS


def _window_mask(t_ms: np.ndarray, start_ms: float, end_ms: float) -> np.ndarray:
    return (t_ms >= start_ms) & (t_ms <= end_ms)


def _segment_mean(x: np.ndarray, t_ms: np.ndarray, start_ms: float, end_ms: float) -> float:
    mask = _window_mask(t_ms, start_ms, end_ms)
    return float(np.mean(x[mask])) if np.any(mask) else float("nan")


def _segment_min(x: np.ndarray, t_ms: np.ndarray, start_ms: float, end_ms: float) -> float:
    mask = _window_mask(t_ms, start_ms, end_ms)
    return float(np.min(x[mask])) if np.any(mask) else float("nan")


def _segment_max(x: np.ndarray, t_ms: np.ndarray, start_ms: float, end_ms: float) -> float:
    mask = _window_mask(t_ms, start_ms, end_ms)
    return float(np.max(x[mask])) if np.any(mask) else float("nan")


def _find_local_positive_peaks(x: np.ndarray, threshold: float) -> np.ndarray:
    idx = np.where((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]) & (x[1:-1] > threshold))[0] + 1
    return idx


def _width_above_level(x: np.ndarray, t_ms: np.ndarray, peak_idx: int, level: float) -> float:
    left = peak_idx
    while left > 0 and x[left] >= level:
        left -= 1
    right = peak_idx
    while right < (x.size - 1) and x[right] >= level:
        right += 1
    return float(t_ms[right] - t_ms[left])


def _recording_duration_ms(t_ms: np.ndarray) -> float:
    if t_ms.size == 0:
        return 0.0
    if t_ms.size == 1:
        return float(t_ms[0])
    dt_ms = float(t_ms[1] - t_ms[0])
    return float(t_ms[-1] + dt_ms)


def _scale_repolarization_time(time_ms: float, duration_ms: float) -> float:
    if time_ms <= REPOLARIZATION_PIVOT_MS:
        return float(time_ms)

    reference_span = REFERENCE_DURATION_MS - REPOLARIZATION_PIVOT_MS
    duration_span = max(duration_ms - REPOLARIZATION_PIVOT_MS, 1.0)
    scale = duration_span / reference_span
    return float(REPOLARIZATION_PIVOT_MS + (time_ms - REPOLARIZATION_PIVOT_MS) * scale)


def _scaled_window(duration_ms: float, start_ms: float, end_ms: float) -> Tuple[float, float]:
    return (
        _scale_repolarization_time(start_ms, duration_ms),
        _scale_repolarization_time(end_ms, duration_ms),
    )


def _validate_rbbb(x: np.ndarray, t_ms: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    RBBB when:
    - two positive QRS peaks in 240-470 ms, separated by at least 28 ms
    - first peak > 0.12
    - second peak > 0.20
    - dip between them < -0.05
    - second peak time > 345 ms
    - terminal width above 0.10 lasting > 38 ms
    """
    qrs_mask = _window_mask(t_ms, 240, 470)
    qrs_idx = np.where(qrs_mask)[0]
    qrs = x[qrs_idx]
    peaks_local = _find_local_positive_peaks(qrs, threshold=0.10)
    if peaks_local.size < 2:
        return False, {"reason": -1.0}

    peaks_idx = qrs_idx[peaks_local]
    best_pair = None
    best_score = -np.inf
    for i in peaks_idx:
        for j in peaks_idx:
            if j <= i:
                continue
            dt = t_ms[j] - t_ms[i]
            if dt < 28:
                continue
            score = x[i] + x[j]
            if score > best_score:
                best_score = score
                best_pair = (i, j)

    if best_pair is None:
        return False, {"reason": -2.0}

    i, j = best_pair
    dip = float(np.min(x[i : j + 1]))
    second_peak_time = float(t_ms[j])
    terminal_width_ms = _width_above_level(x, t_ms, j, level=0.10)

    valid = (
        x[i] > 0.12
        and x[j] > 0.20
        and dip < -0.05
        and second_peak_time > 345.0
        and terminal_width_ms > 38.0
    )
    return valid, {
        "first_peak": float(x[i]),
        "second_peak": float(x[j]),
        "dip_between": dip,
        "second_peak_time_ms": second_peak_time,
        "terminal_width_ms": terminal_width_ms,
    }


def _validate_st_elevation(x: np.ndarray, t_ms: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    ST elevation when:
    - qrs_max > 0.28
    - qrs_min < -0.14
    - mean ST segment from 420-580 ms > 0.09
    - J-point mean from 392-420 ms > 0.09
    """
    duration_ms = _recording_duration_ms(t_ms)
    j_start, j_end = _scaled_window(duration_ms, 392, 420)
    st_start, st_end = _scaled_window(duration_ms, 420, 580)
    qrs_max = _segment_max(x, t_ms, 250, 390)
    qrs_min = _segment_min(x, t_ms, 250, 390)
    st_mean = _segment_mean(x, t_ms, st_start, st_end)
    j_amp = _segment_mean(x, t_ms, j_start, j_end)
    valid = qrs_max > 0.28 and qrs_min < -0.14 and st_mean > 0.09 and j_amp > 0.09
    return valid, {
        "qrs_max": qrs_max,
        "qrs_min": qrs_min,
        "st_mean": st_mean,
        "j_amp_mean": j_amp,
    }


def _validate_t_wave_inversion(
    x: np.ndarray, t_ms: np.ndarray, dt_ms: float
) -> Tuple[bool, Dict[str, float]]:
    """
    T wave inversion when:
    - qrs_max > 0.22
    - rprime_peak > 0.08
    - st_mean > -0.03
    - T-wave minimum < -0.12
    - T-wave minimum occurs after 640 ms, and negative T duration below -0.05 lasts > 45 ms
    """
    duration_ms = _recording_duration_ms(t_ms)
    st_start, st_end = _scaled_window(duration_ms, 420, 560)
    t_start, t_end = _scaled_window(duration_ms, 560, 860)
    t_min_time_threshold = _scale_repolarization_time(640.0, duration_ms)
    qrs_max = _segment_max(x, t_ms, 250, 390)
    t_min = _segment_min(x, t_ms, t_start, t_end)
    st_mean = _segment_mean(x, t_ms, st_start, st_end)
    rprime_peak = _segment_max(x, t_ms, 345, 430)
    t_mask = _window_mask(t_ms, t_start, t_end)
    neg_duration_ms = float(np.sum(x[t_mask] < -0.05) * dt_ms)

    tw_idx = np.where(t_mask)[0]
    t_min_time = float(t_ms[tw_idx[np.argmin(x[tw_idx])]]) if tw_idx.size else float("nan")

    valid = (
        qrs_max > 0.22
        and rprime_peak > 0.08
        and st_mean > -0.03
        and t_min < -0.12
        and t_min_time > t_min_time_threshold
        and neg_duration_ms > 45.0
    )
    return valid, {
        "qrs_max": qrs_max,
        "rprime_peak": rprime_peak,
        "st_mean": st_mean,
        "t_min": t_min,
        "t_min_time_ms": t_min_time,
        "neg_t_duration_ms": neg_duration_ms,
    }


def _validate_normal(
    x: np.ndarray, t_ms: np.ndarray, dt_ms: float
) -> Tuple[bool, Dict[str, float]]:
    """
    Normal beat when:
    - qrs_max > 0.22
    - qrs_min < -0.10
    - st_mean between -0.04 and 0.05
    - t_max > 0.12
    - rprime_peak < 0.10
    """
    duration_ms = _recording_duration_ms(t_ms)
    st_start, st_end = _scaled_window(duration_ms, 420, 580)
    t_start, t_end = _scaled_window(duration_ms, 600, 860)
    qrs_max = _segment_max(x, t_ms, 250, 390)
    qrs_min = _segment_min(x, t_ms, 250, 390)
    st_mean = _segment_mean(x, t_ms, st_start, st_end)
    t_max = _segment_max(x, t_ms, t_start, t_end)
    rprime_peak = _segment_max(x, t_ms, 345, 430)

    has_rbbb = _validate_rbbb(x, t_ms)[0]
    has_st = _validate_st_elevation(x, t_ms)[0]
    has_twi = _validate_t_wave_inversion(x, t_ms, dt_ms)[0]
    valid = (
        qrs_max > 0.22
        and qrs_min < -0.10
        and -0.04 < st_mean < 0.05
        and t_max > 0.12
        and rprime_peak < 0.10
        and not has_rbbb
        and not has_st
        and not has_twi
    )
    return valid, {
        "qrs_max": qrs_max,
        "qrs_min": qrs_min,
        "st_mean": st_mean,
        "t_max": t_max,
        "rprime_peak": rprime_peak,
    }


def _validate_brugada(x: np.ndarray, t_ms: np.ndarray) -> Tuple[bool, Dict[str, float]]:
    """
    Brugada syndrome when:
    - qrs_max > 0.18
    - qrs_min < -0.16
    - j_amp > 0.09
    - st_mean > 0.08
    - coved_cond
    - t_min < -0.08
    """
    duration_ms = _recording_duration_ms(t_ms)
    j_start, j_end = _scaled_window(duration_ms, 392, 420)
    st_start, st_end = _scaled_window(duration_ms, 420, 580)
    t_start, t_end = _scaled_window(duration_ms, 600, 860)
    early_start, early_end = _scaled_window(duration_ms, 430, 500)
    late_start, late_end = _scaled_window(duration_ms, 520, 620)
    j_amp = _segment_mean(x, t_ms, j_start, j_end)
    st_mean = _segment_mean(x, t_ms, st_start, st_end)
    t_min = _segment_min(x, t_ms, t_start, t_end)
    qrs_max = _segment_max(x, t_ms, 250, 390)
    qrs_min = _segment_min(x, t_ms, 250, 390)
    early = _segment_mean(x, t_ms, early_start, early_end)
    late = _segment_mean(x, t_ms, late_start, late_end)
    coved_cond = early > late + 0.03
    valid = (
        qrs_max > 0.18
        and qrs_min < -0.16
        and j_amp > 0.09
        and st_mean > 0.08
        and coved_cond
        and t_min < -0.08
    )
    return valid, {
        "qrs_max": qrs_max,
        "qrs_min": qrs_min,
        "j_amp_mean": j_amp,
        "st_mean": st_mean,
        "t_min": t_min,
        "st_early_mean": early,
        "st_late_mean": late,
    }


def extract_feature_labels(waveform: np.ndarray, fs: int) -> Dict[str, int]:
    t_ms = np.arange(waveform.shape[0], dtype=float) / fs * 1000.0
    dt_ms = 1000.0 / fs
    return {
        "RBBB": int(_validate_rbbb(waveform, t_ms)[0]),
        "ST_ELEVATION": int(_validate_st_elevation(waveform, t_ms)[0]),
        "T_WAVE_INVERSION": int(_validate_t_wave_inversion(waveform, t_ms, dt_ms)[0]),
    }


def _validate_beat(
    class_key: str, x: np.ndarray, t_ms: np.ndarray, fs: int
) -> Tuple[bool, Dict[str, float]]:
    dt_ms = 1000.0 / fs
    if class_key == "NORMAL":
        return _validate_normal(x, t_ms, dt_ms)
    if class_key == "RBBB":
        return _validate_rbbb(x, t_ms)
    if class_key == "ST_ELEVATION":
        return _validate_st_elevation(x, t_ms)
    if class_key == "T_WAVE_INVERSION":
        return _validate_t_wave_inversion(x, t_ms, dt_ms)
    if class_key == "BRUGADA":
        return _validate_brugada(x, t_ms)
    raise ValueError(f"Unsupported class key: {class_key}")


__all__ = [
    "_window_mask",
    "_segment_mean",
    "_segment_min",
    "_segment_max",
    "_find_local_positive_peaks",
    "_width_above_level",
    "_validate_rbbb",
    "_validate_st_elevation",
    "_validate_t_wave_inversion",
    "_validate_normal",
    "_validate_brugada",
    "extract_feature_labels",
    "_validate_beat",
]