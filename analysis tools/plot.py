import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np
# Load WAV file
sample_rate, data = wavfile.read("dronesound_hrtf_snippets_real/az0_el0_snippet013.wav")

# If stereo, take just one channel
if data.ndim > 1:
    data = data[:, 0]

# Create time axis in seconds
time = np.arange(len(data)) / sample_rate

# Plot waveform
plt.figure(figsize=(12, 4))
plt.plot(time, data, color='blue')
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.title("Waveform of your_file.wav")
plt.grid(True)
plt.show()