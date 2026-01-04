import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from scipy.io import wavfile
from scipy.ndimage import zoom
from numba import njit, prange
import matplotlib.pyplot as plt
from PIL import Image
import time

# ============================================================
# TIMING HELPER
# ============================================================
def stamp(msg, t0):
    t = time.perf_counter()
    print(f"[{t - t0:8.3f} s] {msg}")
    return t

# ============================================================
# NUMBA KERNEL (FAST INTERVAL FILL)
# ============================================================
@njit(parallel=True, fastmath=True)
def fill_amplitudes_numba_fast(all_times, all_amps, fs, num_samples):
    nrows = len(all_times)

    amp = np.zeros((nrows, num_samples), np.float64)
    bw  = np.zeros((nrows, num_samples), np.uint8)

    for i in prange(nrows):
        times = all_times[i]
        vals  = all_amps[i]

        if len(times) == 0:
            continue

        idxs = np.round(times * fs).astype(np.int64)

        for j in range(len(idxs)):
            if idxs[j] >= num_samples:
                idxs[j] = num_samples - 1

        last = 0
        last_val = 0.0

        for j in range(len(idxs)):
            idx = idxs[j]

            if idx > last:
                amp[i, last:idx] = -last_val

            amp[i, idx] = vals[j]
            bw[i, idx] = 1

            last_val = vals[j]
            last = idx + 1

        if last < num_samples:
            amp[i, last:] = -last_val

    return amp, bw

# ============================================================
# PARAMETERS
# ============================================================
order = 2
bandwidth = 40
freqs = np.arange(800, 5001, 5)
fs_analysis = 340 / 0.01   # 17000 Hz

# ============================================================
# ANALYSIS FUNCTION
# ============================================================
def analyze_channel(signal, orig_fs):
    nyq = orig_fs / 2

    # ---- Precompute filters once
    filters = []
    for f in freqs:
        if f + bandwidth >= nyq:
            break
        filters.append(
            butter(order, [f / nyq, (f + bandwidth) / nyq], btype="band")
        )

    t = np.arange(len(signal)) / orig_fs

    all_times = []
    all_amps  = []

    for b, a in filters:
        y = lfilter(b, a, signal)
        peaks, _ = find_peaks(y)

        all_times.append(t[peaks])
        all_amps.append(y[peaks])

    # ---- Compute required length
    max_time = max((row.max() for row in all_times if len(row)), default=0.0)
    num_samples = int(np.ceil(max_time * fs_analysis)) + 1

    # ---- Convert to arrays for Numba
    all_times = [np.asarray(r, np.float64) for r in all_times]
    all_amps  = [np.asarray(r, np.float64) for r in all_amps]

    return fill_amplitudes_numba_fast(
        all_times, all_amps, fs_analysis, num_samples
    )

# ============================================================
# MAIN
# ============================================================
t0 = time.perf_counter()
print("=== START ===")
# Warm-up call with small dummy arrays
dummy_times = [np.array([0.0])]
dummy_amps  = [np.array([0.0])]
dummy_num_samples = 10
dummy_fs = 1.0

_ = fill_amplitudes_numba_fast(dummy_times, dummy_amps, dummy_fs, dummy_num_samples)
t0 = stamp("numba warmup call", t0)
# ---- Load WAV
sr, data = wavfile.read("accelerate.wav")
t0 = stamp("Loaded WAV file", t0)

# ---- Prepare channels
if data.ndim == 1:
    data = np.stack([data, data], axis=1)

left  = data[:, 0].astype(np.float64)
right = data[:, 1].astype(np.float64)
t0 = stamp("Prepared channels", t0)

# ---- Analyze left
amp_left, amp_left_bw = analyze_channel(left, sr)
t0 = stamp("Analyzed LEFT channel", t0)

# ---- Analyze right
amp_right, amp_right_bw = analyze_channel(right, sr)
t0 = stamp("Analyzed RIGHT channel", t0)

# ---- Crop binary images
amp_left_bw_short  = amp_left_bw[:, 1650:1750]
amp_right_bw_short = amp_right_bw[:, 1650:1750]
t0 = stamp("Cropped BW images", t0)

# ---- Resize continuous images (PIL)
amp_left  = Image.fromarray(amp_left)
amp_right = Image.fromarray(amp_right)

amp_left_long  = amp_left.resize((340, 84), Image.BILINEAR)
amp_right_long = amp_right.resize((340, 84), Image.BILINEAR)
t0 = stamp("Resized amplitude images", t0)

# ---- Convert back to NumPy
amp_left_long  = np.array(amp_left_long)
amp_right_long = np.array(amp_right_long)
t0 = stamp("Converted resized images to NumPy", t0)

# ---- Save images
plt.imsave("out_left_ai.png",  amp_left_long,  cmap="binary", origin="lower",
           vmin=-2000, vmax=2000)
plt.imsave("out_right_ai.png", amp_right_long, cmap="binary", origin="lower",
           vmin=-2000, vmax=2000)

plt.imsave("out_left_bw_ai.png",  amp_left_bw_short,  cmap="binary", origin="lower",
           vmin=0, vmax=1)
plt.imsave("out_right_bw_ai.png", amp_right_bw_short, cmap="binary", origin="lower",
           vmin=0, vmax=1)
t0 = stamp("Saved images", t0)

print("=== DONE ===")