import os
import re
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
from scipy.io import wavfile
import matplotlib.pyplot as plt
from numba import njit, prange


@njit(parallel=True)
def fill_amplitudes_numba(all_max_times, all_max_amplitudes, fs, num_samples):
    num_rows = len(all_max_times)
    amplitudes_2dr = np.zeros((num_rows, num_samples), dtype=np.float64)
    amplitudes_bw = np.zeros((num_rows, num_samples), dtype=np.float64)
    
    for i in prange(num_rows):  # parallel over rows
        times_row = all_max_times[i]
        amps_row = all_max_amplitudes[i]
        last_val = 0.0
        
        # Precompute indices
        idxs = np.empty(len(times_row), dtype=np.int64)
        for j in range(len(times_row)):
            idx = int(round(times_row[j] * fs))
            if idx >= num_samples:
                idx = num_samples - 1
            idxs[j] = idx
        
        amp_dict = dict()
        for j in range(len(idxs)):
            amp_dict[idxs[j]] = amps_row[j]

        # Fill row sequentially
        for t in range(num_samples):
            if t in amp_dict:
                last_val = amp_dict[t]
                amplitudes_2dr[i, t] = last_val
                amplitudes_bw[i, t] = 1
            else:
                amplitudes_2dr[i, t] = -last_val
                amplitudes_bw[i, t] = 0

    return amplitudes_2dr,amplitudes_bw


@njit(parallel=True)
def fill_amplitudes_numba_top20(all_max_times, all_max_amplitudes, fs, num_samples):
    num_rows = len(all_max_times)
    amplitudes_2dr = np.zeros((num_rows, num_samples), dtype=np.float64)
    amplitudes_bw = np.zeros((num_rows, num_samples), dtype=np.float64)

    # Step 1: Precompute indices and flatten all amplitudes for top 20% threshold
    all_flat_amps = []

    # store row-wise indices and amplitudes for later
    idx_list = [np.empty(len(all_max_times[i]), dtype=np.int64) for i in range(num_rows)]
    amp_list = [np.empty(len(all_max_times[i]), dtype=np.float64) for i in range(num_rows)]

    for i in range(num_rows):
        times_row = all_max_times[i]
        amps_row = all_max_amplitudes[i]

        idxs = idx_list[i]
        amps = amp_list[i]
        for j in range(len(times_row)):
            idx = int(round(times_row[j] * fs))
            if idx >= num_samples:
                idx = num_samples - 1
            idxs[j] = idx
            amps[j] = amps_row[j]
            all_flat_amps.append(amps_row[j])

    # Step 2: compute threshold for top 20%
    threshold = np.percentile(np.array(all_flat_amps), 90)  # top 20%

    # Step 3: Fill the arrays
    for i in prange(num_rows):
        last_val = 0.0
        idxs = idx_list[i]
        amps = amp_list[i]

        # build dict for convenience
        amp_dict = dict()
        for j in range(len(idxs)):
            amp_dict[idxs[j]] = amps[j]

        for t in range(num_samples):
            if t in amp_dict:
                last_val = amp_dict[t]
                amplitudes_2dr[i, t] = last_val
                # only set 1 if amp is in top 20% of all values
                if last_val >= threshold:
                    amplitudes_bw[i, t] = 1
                else:
                    amplitudes_bw[i, t] = 0
            else:
                amplitudes_2dr[i, t] = -last_val
                amplitudes_bw[i, t] = 0

    return amplitudes_2dr, amplitudes_bw

# -----------------------------
# Parameters
# -----------------------------
#input_folder = "dronesound_hrtf_snippets"   # WAV snippets
#output_folder = "plots_left_right_snippets_el0_bin" # PNGs here
#input_folder = "dronesound_hrtf_snippets_real"   # WAV snippets
#output_folder = "plots_left_right_snippets_el0_bin_real" # PNGs here
input_folder = "dronesound_hrtf_snippets_real1"   # WAV snippets
output_folder = "plots_left_right_snippets_el0_bin_real1" # PNGs here
os.makedirs(output_folder, exist_ok=True)

order = 2
bandwidth = 40
freqs = np.arange(800, 5001, 5)
threshold = 0
fs_analysis = 340 / 0.01  # 17,000 Hz for 2cm resolution


# -----------------------------
# Analysis function
# -----------------------------
def analyze_channel(signal, orig_fs):
    all_max_times = []
    all_max_amplitudes = []

    t = np.linspace(0, len(signal)/orig_fs, len(signal), endpoint=False)

    for f_start in freqs:
        f_low = f_start
        f_high = f_start + bandwidth
        if f_high >= orig_fs / 2:
            break

        nyq = orig_fs / 2
        b, a = butter(order, [f_low/nyq, f_high/nyq], btype='band')
        y = lfilter(b, a, signal)*4

        peaks, _ = find_peaks(y)
        all_max_times.append(t[peaks])
        all_max_amplitudes.append(y[peaks])

    max_time = max([max(row) if len(row) > 0 else 0 for row in all_max_times])
    num_samples = int(np.ceil(max_time * fs_analysis)) + 1

    return fill_amplitudes_numba(all_max_times, all_max_amplitudes, fs_analysis, num_samples)


# -----------------------------
# Regex to extract azimuth, elevation, snippet
# -----------------------------
pattern = re.compile(r'az(-?\d+)_el(-?\d+)_snippet(\d+)')


# -----------------------------
# Process each WAV snippet
# -----------------------------
for wav_file in os.listdir(input_folder):
    if not wav_file.endswith(".wav"):
        continue

    match = pattern.search(wav_file)
    if match:
        azimuth = int(match.group(1))
        elevation = int(match.group(2))
        snippet_idx = int(match.group(3))
    else:
        continue  # skip files not matching the pattern

    # ------------------------------------------------------
    # 🔥 Filter: ONLY elevation 0
    if elevation != 0:
        continue

    # 🔥 Filter: ONLY snippet 34
    #if snippet_idx != 34:
    #    continue
    # ------------------------------------------------------

    filepath = os.path.join(input_folder, wav_file)
    sr, data = wavfile.read(filepath)

    # Ensure stereo
    if data.ndim == 1:
        data = np.stack([data, data], axis=1)

    left = data[:, 0]# / 32768.0
    right = data[:, 1]# / 32768.0

    amp_left,amp_leftbw = analyze_channel(left, sr)
    amp_right,amp_rightbw = analyze_channel(right, sr)

    out_left = os.path.join(
        output_folder,
        f"az{azimuth}_el{elevation}_snippet{snippet_idx:03d}_left.png"
    )
    out_right = os.path.join(
        output_folder,
        f"az{azimuth}_el{elevation}_snippet{snippet_idx:03d}_right.png"
    )

    out_left_bw = os.path.join(
        output_folder,
        f"az{azimuth}_el{elevation}_snippet{snippet_idx:03d}_left_bw.png"
    )
    out_right_bw = os.path.join(
        output_folder,
        f"az{azimuth}_el{elevation}_snippet{snippet_idx:03d}_right_bw.png"
    )

    vmin = min(amp_left.min(), amp_left.min())
    vmax = max(amp_right.max(), amp_right.max())
    print(vmin)
    print(vmax)
    vmax = 2000
    vmin = -2000
    
    vmax_bw = 1
    vmin_bw = 0

    plt.imsave(out_left, amp_left, cmap='binary', origin='lower', vmin=vmin, vmax=vmax)
    plt.imsave(out_right, amp_right, cmap='binary', origin='lower', vmin=vmin, vmax=vmax)

    plt.imsave(out_left_bw, amp_leftbw, cmap='binary', origin='lower', vmin=vmin_bw, vmax=vmax_bw)
    plt.imsave(out_right_bw, amp_rightbw, cmap='binary', origin='lower', vmin=vmin_bw, vmax=vmax_bw)

print("Saved plots for elevation 0, snippet 34 only.")