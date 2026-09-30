import numpy as np
import pytest
from scipy.signal import lfilter

from src import config as cfg
from src import preprocess as pp

SR_IN = 44100


def tone(freq, sr, dur=1.0, amp=0.5):
    n = np.arange(int(sr * dur))
    return amp * np.sin(2 * np.pi * freq * n / sr)


def power_at(x, sr, freq):
    X = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return X[np.argmin(np.abs(f - freq))]


def test_resample_length_and_frequency():
    y = pp.resample(tone(1000, SR_IN), SR_IN, cfg.SR)
    assert len(y) == cfg.SR                                  # 1 s stays 1 s
    f = np.fft.rfftfreq(len(y), 1 / cfg.SR)
    assert f[np.argmax(np.abs(np.fft.rfft(y)))] == pytest.approx(1000, abs=1)


def test_resample_removes_content_above_new_nyquist():
    """10 kHz is above 8 kHz: the anti-aliasing filter must remove it,
    otherwise it would fold back to 16 - 10 = 6 kHz.
    scipy's default Kaiser (beta = 5) measures about -57 dB here."""
    y_hi = pp.resample(tone(10_000, SR_IN), SR_IN, cfg.SR)
    y_ref = pp.resample(tone(1000, SR_IN), SR_IN, cfg.SR)
    alias = power_at(y_hi, cfg.SR, 6000)
    ref = power_at(y_ref, cfg.SR, 1000)
    assert 20 * np.log10(alias / ref) < -50


def test_resample_keeps_passband():
    """7 kHz (below the new 8 kHz Nyquist) must pass almost unchanged."""
    y_pass = pp.resample(tone(7000, SR_IN), SR_IN, cfg.SR)
    y_ref = pp.resample(tone(1000, SR_IN), SR_IN, cfg.SR)
    assert abs(20 * np.log10(power_at(y_pass, cfg.SR, 7000) / power_at(y_ref, cfg.SR, 1000))) < 1


def test_to_mono():
    x = np.random.default_rng(0).normal(size=100)
    np.testing.assert_allclose(pp.to_mono(np.stack([x, x], axis=1)), x)
    np.testing.assert_allclose(pp.to_mono(x), x)


def test_pre_emphasis_matches_difference_equation():
    x = np.random.default_rng(1).normal(size=500)
    y = pp.pre_emphasis(x, 0.97)
    expected = np.concatenate([[x[0]], x[1:] - 0.97 * x[:-1]])
    np.testing.assert_allclose(y, expected)
    np.testing.assert_allclose(y, lfilter([1, -0.97], [1], x))


def test_trim_silence_cuts_edges_keeps_middle_pause():
    sr = cfg.SR
    speech = tone(200, sr, 0.5)
    gap = np.zeros(int(0.3 * sr))
    x = np.concatenate([np.zeros(sr), speech, gap, speech, np.zeros(sr)])
    y = pp.trim_silence(x, sr)
    assert len(y) == pytest.approx(len(speech) * 2 + len(gap), abs=int(0.05 * sr))
    assert np.max(np.abs(y[: int(0.02 * sr)])) > 0          # starts at speech


def test_trim_silence_keeps_signal_if_result_too_short():
    x = np.concatenate([np.zeros(5000), tone(200, cfg.SR, 0.01), np.zeros(5000)])
    np.testing.assert_array_equal(pp.trim_silence(x, cfg.SR), x)


def test_trim_silence_all_zero():
    x = np.zeros(4000)
    np.testing.assert_array_equal(pp.trim_silence(x, cfg.SR), x)


def test_frame_signal_covers_every_sample():
    x = np.arange(3000, dtype=float)
    frames = pp.frame_signal(x, 1024, 512)
    assert frames.shape[1] == 1024
    assert frames[-1].max() == x[-1] or x[-1] in frames[-1]
    np.testing.assert_array_equal(frames[1][:512], x[512:1024])


def test_frame_signal_short_input_gives_one_padded_frame():
    frames = pp.frame_signal(np.ones(300), 1024, 512)
    assert frames.shape == (1, 1024)
    assert frames[0, :300].sum() == 300 and frames[0, 300:].sum() == 0
