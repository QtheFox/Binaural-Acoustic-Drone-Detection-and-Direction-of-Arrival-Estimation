import os
import re
import numpy as np
from scipy.signal import butter, lfilter, find_peaks, correlate
from scipy.io import wavfile
from numba import njit, prange
from PIL import Image

# ============================================================
# NUMBA: amplitude filling
# ============================================================

@njit(parallel=True)
def fill_amplitudes_numba(all_max_times, all_max_amplitudes, fs, num_samples):
    num_rows = len(all_max_times)
    amplitudes_2dr = np.zeros((num_rows, num_samples), dtype=np.float64)

    for i in prange(num_rows):
        times_row = all_max_times[i]
        amps_row = all_max_amplitudes[i]
        last_val = 0.0

        idxs = np.empty(len(times_row), dtype=np.int64)
        for j in range(len(times_row)):
            idx = int(round(times_row[j] * fs))
            if idx >= num_samples:
                idx = num_samples - 1
            idxs[j] = idx

        amp_dict = dict()
        for j in range(len(idxs)):
            amp_dict[idxs[j]] = amps_row[j]

        for t in range(num_samples):
            if t in amp_dict:
                last_val = amp_dict[t]
                amplitudes_2dr[i, t] = last_val
            else:
                amplitudes_2dr[i, t] = -last_val

    return amplitudes_2dr


# ============================================================
# PARAMETERS
# ============================================================

input_folder = "dronesound_hrtf_snippets_real1"
#input_folder = "dataset_debug"
output_folder = "dataset_real2"
os.makedirs(output_folder, exist_ok=True)

order = 2
bandwidth = 40
freqs = np.arange(800, 5001, 5)

fs_analysis = 340 / 0.01  # 17 kHz

# correlogram parameters
fs_target = fs_analysis
max_delay_ms = 1.4

pattern = re.compile(r'az(-?\d+)_el(-?\d+)_snippet(\d+)')


# ============================================================
# ANALYSIS FUNCTION
# ============================================================

def analyze_channel_with_fb(signal, orig_fs):
    all_max_times = []
    all_max_amplitudes = []
    filterbank_outputs = []

    t = np.linspace(0, len(signal)/orig_fs, len(signal), endpoint=False)

    for f_start in freqs:
        f_low = f_start
        f_high = f_start + bandwidth
        if f_high >= orig_fs / 2:
            break

        nyq = orig_fs / 2
        b, a = butter(order, [f_low/nyq, f_high/nyq], btype='band')
        y = lfilter(b, a, signal) * 4

        filterbank_outputs.append(y)

        peaks, _ = find_peaks(y)
        all_max_times.append(t[peaks])
        all_max_amplitudes.append(y[peaks])

    max_time = max([max(row) if len(row) > 0 else 0 for row in all_max_times])
    num_samples = int(np.ceil(max_time * fs_analysis)) + 1

    amp = fill_amplitudes_numba(
        all_max_times, all_max_amplitudes, fs_analysis, num_samples
    )

    return amp, np.array(filterbank_outputs)


# ============================================================
# BIN AURAL CORRELOGRAM
# ============================================================

def compute_correlogram(left_fb, right_fb, fs_target, max_delay_ms):
    max_lag = int(fs_target * max_delay_ms / 1000)
    correlogram = []

    for b in range(left_fb.shape[0]):
        corr = correlate(left_fb[b], right_fb[b], mode="full")
        mid = len(corr) // 2
        corr = corr[mid - max_lag : mid + max_lag]

        norm = np.sqrt(
            np.sum(left_fb[b]**2) * np.sum(right_fb[b]**2)
        ) + 1e-12

        corr /= norm
        correlogram.append(corr)

    return np.array(correlogram)


# ============================================================
# NUMPY → PIL IMAGE
# ============================================================

def to_img(arr, vmin, vmax):
    # Clip and normalize
    arr = np.clip(arr, vmin, vmax)
    arr = (arr - vmin) / (vmax - vmin)
    # Invert grayscale (like matplotlib 'binary')
    arr = 1.0 - arr
    arr = (arr * 255).astype(np.uint8)
    # Flip vertically to match origin='lower'
    arr = np.flipud(arr)
    return Image.fromarray(arr)


# ============================================================
# MAIN PIPELINE (PROCESS ELEVATIONS IN ORDER)
# ============================================================

# Collect all files with az/el/snippet info
wav_files_info = []

for wav_file in os.listdir(input_folder):
    if not wav_file.endswith(".wav"):
        continue

    match = pattern.search(wav_file)
    if not match:
        continue

    az, el, snippet = map(int, match.groups())
    wav_files_info.append((el, az, snippet, wav_file))

# Sort by elevation first (0, 12, 24, ...) then azimuth, then snippet
wav_files_info.sort(key=lambda x: (x[0], x[1], x[2]))

for el, az, snippet, wav_file in wav_files_info:
    sr, data = wavfile.read(os.path.join(input_folder, wav_file))
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)

    left = data[:, 0]
    right = data[:, 1]

    # ---- analysis ----
    amp_l, fb_l = analyze_channel_with_fb(left, sr)
    amp_r, fb_r = analyze_channel_with_fb(right, sr)

    corr = compute_correlogram(fb_l, fb_r, fs_target, max_delay_ms)

    # ---- convert to images ----
    img_left = to_img(amp_l, -2000, 2000).resize((340, 84))
    img_right = to_img(amp_r, -2000, 2000).resize((340, 84))
    img_corr = to_img(corr, -1.0, 1.0).resize((340, 84))

    # ---- save dataset ----
    snippet_dir = os.path.join(
        output_folder, f"Az{az}_El{el}", f"{snippet:03d}"
    )
    os.makedirs(snippet_dir, exist_ok=True)

    img_left.save(os.path.join(snippet_dir, "left.png"))
    img_right.save(os.path.join(snippet_dir, "right.png"))
    img_corr.save(os.path.join(snippet_dir, "correlogram.png"))

print("✅ Dataset with correlogram generated in elevation order (0°, 12°, 24°, ...).")
