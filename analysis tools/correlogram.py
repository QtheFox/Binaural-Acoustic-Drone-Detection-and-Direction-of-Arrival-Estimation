import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp, butter, lfilter, correlate
import pyroomacoustics as pra
import soundfile as sf

# ======================================================
# GLOBAL PARAMETERS
# ======================================================
fs = 44100
duration = 0.2
c = 343
ear_distance = 0.18

# Filter bank parameters (yours)
order = 2
bandwidth = 40
freqs = np.arange(800, 5001, 5)
fs_analysis = 340 / 0.01   # 17000 Hz (conceptual resolution)

# ======================================================
# 1) GENERATE CHIRP
# ======================================================
t = np.linspace(0, duration, int(fs * duration), endpoint=False)
# Choose the signal type: "chirp", "random", or "sines"
signal_type = "random"

if signal_type == "chirp":
    # Linear chirp from 500 Hz to 4000 Hz
    signal = chirp(t, f0=500, f1=4000, t1=duration, method="linear")
    
elif signal_type == "random":
    # White noise
    signal = np.random.randn(len(t))
    
elif signal_type == "sines":
    # Sum of three sine waves: 500 Hz, 1000 Hz, 2000 Hz
    signal = (np.sin(2*np.pi*500*t) +
              np.sin(2*np.pi*1000*t) +
              np.sin(2*np.pi*2000*t))
    # Optional: normalize to -1..1
    signal /= np.max(np.abs(signal))
    
else:
    raise ValueError("Unknown signal_type. Choose 'chirp', 'random', or 'sines'.")

# ======================================================
# 2) ROOM SIMULATION
# ======================================================
room = pra.ShoeBox(
    [6, 5, 3],
    fs=fs,
    absorption=0.4,
    max_order=10
)

room.add_source([3, 2, 1.5], signal=signal)

mic = np.array([[3], [2.5], [1.5]])
room.add_microphone_array(pra.MicrophoneArray(mic, fs))

room.simulate()
room_signal = room.mic_array.signals[0]

# ======================================================
# 3) SIMPLE HRTF MODEL (ITD + ILD)
# ======================================================
azimuth_deg = 30.0
azimuth_rad = np.deg2rad(azimuth_deg)

itd = (ear_distance / c) * np.sin(azimuth_rad)
sample_delay = int(itd * fs)

ild = np.cos(azimuth_rad)

if sample_delay >= 0:
    left = room_signal
    right = np.pad(room_signal, (sample_delay, 0))[:len(room_signal)]
else:
    left = np.pad(room_signal, (-sample_delay, 0))[:len(room_signal)]
    right = room_signal

left *= ild
right *= (2 - ild)

binaural = np.column_stack((left, right))
sf.write("chirp_room_binaural.wav", binaural, fs)




# ======================================================
# 4) FILTER BANK (NO PEAK EXTRACTION)
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

        b, a = butter(
            order,
            [f_low / nyq, f_high / nyq],
            btype="band"
        )

        y = lfilter(b, a, signal)
        filtered.append(y)
        center_freqs.append((f_low + f_high) / 2)

    return np.array(filtered), np.array(center_freqs)

left_fb, fb_freqs = run_filter_bank(left, fs)
right_fb, _ = run_filter_bank(right, fs)

# ======================================================
# 5) OPTIONAL: BINAURAL CORRELOGRAM PER BAND
# ======================================================
max_delay_ms = 1.0
max_lag = int(fs * max_delay_ms / 1000)

correlogram = []

for band in range(left_fb.shape[0]):
    corr = correlate(left_fb[band], right_fb[band], mode="full")
    mid = len(corr) // 2
    corr = corr[mid - max_lag : mid + max_lag]

    corr /= np.sqrt(
        np.sum(left_fb[band]**2) * np.sum(right_fb[band]**2) + 1e-12
    )

    correlogram.append(corr)

correlogram = np.array(correlogram)

# ======================================================
# 6) VISUALIZATION
# ======================================================
lags = np.linspace(-max_delay_ms, max_delay_ms, correlogram.shape[1])

plt.figure(figsize=(10, 6))
plt.imshow(
    correlogram,
    aspect="auto",
    origin="lower",
    extent=[lags[0], lags[-1], fb_freqs[0], fb_freqs[-1]],
    cmap="inferno"
)
plt.xlabel("Delay (ms)")
plt.ylabel("Center Frequency (Hz)")
plt.title("Binaural Correlogram (Filter Bank Output)")
plt.colorbar(label="Correlation")
plt.tight_layout()
#plt.show()




def add_with_fractional_shift(left_fb, right_fb, fs, shift_ms=0.0):
    """
    Adds left and right filter bank signals with a fractional shift in milliseconds.

    Parameters:
        left_fb, right_fb : np.array, shape (n_bands, n_samples)
        fs : sampling frequency
        shift_ms : float, time shift in milliseconds (positive shifts right_fb forward, negative backward)

    Returns:
        summed_fb : np.array, shape (n_bands, n_samples)
    """
    n_bands, n_samples = left_fb.shape
    shift_samples = shift_ms * fs / 1000.0  # fractional samples

    summed_fb = []

    for b in range(n_bands):
        # original indices
        original_idx = np.arange(n_samples)
        # shifted indices
        shifted_idx = original_idx - shift_samples

        # linear interpolation
        shifted_right = np.interp(shifted_idx, original_idx, right_fb[b], left=0, right=0)

        summed_fb.append(left_fb[b] + shifted_right)

    return np.array(summed_fb)



# ======================================================
# 7) PLOT LEFT AND SUMMED FILTER BANK (LINEAR, NO dB)
# ======================================================

# Keep linear amplitudes for comparison
left_fb_lin = left_fb
summed_fb_lin = add_with_fractional_shift(left_fb_lin, right_fb, fs, shift_ms=-0.7)

# Compute common color scale
vmin = min(left_fb_lin.min(), summed_fb_lin.min())
vmax = max(left_fb_lin.max(), summed_fb_lin.max())

# Plot left and summed
plt.figure(figsize=(12, 6))

# Left
plt.subplot(2, 1, 1)
plt.imshow(
    left_fb_lin,
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
plt.xlabel("Time (s)")
plt.ylabel("Freq (Hz)")
plt.yscale('log')
plt.title(f"Summed Filter Bank Output (Linear, shift {-0.25} ms)")

# Single colorbar for both subplots
cbar_ax = plt.gcf().add_axes([0.92, 0.15, 0.02, 0.7])  # x, y, width, height
norm = plt.cm.ScalarMappable(cmap='inferno', norm=plt.Normalize(vmin=vmin, vmax=vmax))
norm.set_array([])
plt.colorbar(norm, cax=cbar_ax, label='Amplitude')

plt.tight_layout(rect=[0, 0, 0.9, 1])  # leave space for colorbar
plt.show()

