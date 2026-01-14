import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, correlate
import soundfile as sf

# ======================================================
# GLOBAL PARAMETERS
# ======================================================
fs_target = 44100
order = 2
bandwidth = 40
freqs = np.arange(800, 5001, 5)
max_delay_ms = 2.0  # for correlogram

# ======================================================
# 1) LOAD STEREO AUDIO FILE
# ======================================================
audio, fs_audio = sf.read("dronesound_hrtf_snippets_real1/az204_el0_snippet023.wav")
if fs_audio != fs_target:
    raise ValueError("Sampling rate must match fs_target")

if audio.ndim != 2 or audio.shape[1] < 2:
    raise ValueError("Audio file must be stereo (2 channels)")

left = audio[:, 0]
right = audio[:, 1]

min_len = min(len(left), len(right))
left = left[:min_len]
right = right[:min_len]

duration = len(left) / fs_target
t = np.linspace(0, duration, len(left), endpoint=False)

# ======================================================
# 2) FILTER BANK
# ======================================================
def run_filter_bank(signal, fs):
    nyq = fs / 2
    filtered = []
    center_freqs = []
    for f_start in freqs:
        f_low = f_start
        f_high = f_start + bandwidth
        if f_high >= nyq:
            break
        b, a = butter(order, [f_low/nyq, f_high/nyq], btype="band")
        y = lfilter(b, a, signal)
        filtered.append(y)
        center_freqs.append((f_low + f_high)/2)
    return np.array(filtered), np.array(center_freqs)

left_fb, fb_freqs = run_filter_bank(left, fs_target)
right_fb, _ = run_filter_bank(right, fs_target)

# ======================================================
# 3) FRACTIONAL SHIFT SUM
# ======================================================
def add_with_fractional_shift(left_fb, right_fb, fs, shift_ms=0.0):
    n_bands, n_samples = left_fb.shape
    shift_samples = shift_ms * fs / 1000.0
    summed_fb = []
    for b in range(n_bands):
        original_idx = np.arange(n_samples)
        shifted_idx = original_idx - shift_samples
        shifted_right = np.interp(shifted_idx, original_idx, right_fb[b], left=0, right=0)
        summed_fb.append(left_fb[b] + shifted_right)
    return np.array(summed_fb)

shift_ms = -0.75
summed_fb_lin = add_with_fractional_shift(left_fb, right_fb, fs_target, shift_ms=shift_ms)
# Choose a nonlinear exponent (>1 boosts strong values)
gamma = 4.0  # you can try 1.5, 2.5, etc.


# ======================================================
# 4) BINARUAL CORRELOGRAM PER BAND
# ======================================================
max_lag = int(fs_target * max_delay_ms / 1000)
correlogram = []

for b in range(left_fb.shape[0]):
    corr = correlate(left_fb[b], right_fb[b], mode="full")
    mid = len(corr)//2
    corr = corr[mid - max_lag : mid + max_lag]
    # Normalize correlation
    corr /= np.sqrt(np.sum(left_fb[b]**2) * np.sum(right_fb[b]**2) + 1e-12)
    correlogram.append(corr)

correlogram = np.array(correlogram)
lags = np.linspace(-max_delay_ms, max_delay_ms, correlogram.shape[1])

# ======================================================
# 5) PLOT LEFT, SUMMED, AND CORRELOGRAM
# ======================================================
left_fb = np.abs(left_fb)**gamma
summed_fb_lin = np.abs(summed_fb_lin)**gamma
vmin = min(left_fb.min(), summed_fb_lin.min())
vmax = max(left_fb.max(), summed_fb_lin.max())
vmin = 0
vmax = 0.0000012

plt.figure(figsize=(12, 10))
# Apply nonlinear scaling for visualization

# Left
plt.subplot(2, 1, 1)
plt.imshow(
    left_fb,
    aspect='auto',
    origin='lower',
    extent=[0, duration, fb_freqs[0], fb_freqs[-1]],
    cmap='inferno',
    vmin=vmin, vmax=vmax
)
plt.ylabel("Freq (Hz)")
plt.yscale('log')
plt.title("Left Ear Filter Bank Output (Linear)")

# Summed
plt.subplot(2, 1, 2)
plt.imshow(
    summed_fb_lin,
    aspect='auto',
    origin='lower',
    extent=[0, duration, fb_freqs[0], fb_freqs[-1]],
    cmap='inferno',
    vmin=vmin, vmax=vmax
)
plt.ylabel("Freq (Hz)")
plt.yscale('log')
plt.title(f"Summed Filter Bank Output (Linear, shift {shift_ms} ms)")
# Single colorbar for both subplots
cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # x, y, width, height
norm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=vmin, vmax=vmax))
norm.set_array([])
plt.colorbar(norm, cax=cbar_ax, label='Amplitude')
# Correlogram
plt.figure(figsize=(10, 6))
plt.imshow(
    correlogram,
    aspect='auto',
    origin='lower',
    extent=[lags[0], lags[-1], fb_freqs[0], fb_freqs[-1]],
    cmap='inferno'
)
plt.xlabel("Delay (ms)")
plt.ylabel("Freq (Hz)")
plt.yscale('log')
plt.title("Binaural Correlogram (Normalized)")
plt.colorbar(label="Correlation")

plt.tight_layout()
plt.show()