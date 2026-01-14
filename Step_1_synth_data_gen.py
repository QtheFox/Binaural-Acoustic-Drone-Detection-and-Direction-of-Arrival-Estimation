import os
import slab
import numpy as np
from scipy.io import wavfile

# -----------------------------
# Parameters
# -----------------------------
input_file = "dronesound_cut.wav"
snippet_duration = 0.1  # seconds
fade_ms = 5             # optional fade-in/out
out_folder = "dronesound_hrtf_snippets"
os.makedirs(out_folder, exist_ok=True)

# -----------------------------
# Load original sound
# -----------------------------
sound = slab.Sound(input_file)  # mono preferred
sr = sound.samplerate
num_samples_per_snippet = int(snippet_duration * sr)
fade_samples = int(fade_ms / 1000 * sr)

# -----------------------------
# Compute overall min/max after converting original sound to 16-bit PCM
# -----------------------------
sound_int16 = np.int16(np.clip(sound.data, -1, 1) * 32767)
overall_min = int(sound_int16.min())
overall_max = int(sound_int16.max())
print(f"Overall min (16-bit PCM): {overall_min}, Overall max (16-bit PCM): {overall_max}")

# -----------------------------
# Load KEMAR HRTF
# -----------------------------
hrtf = slab.HRTF.kemar()
num_sources = len(hrtf.sources.vertical_polar)
print(f"KEMAR HRTF loaded with {num_sources} source positions.")

# -----------------------------
# Apply HRTF and split into snippets
# -----------------------------
for idx, (azimuth, elevation, distance) in enumerate(hrtf.sources.vertical_polar):
    # Apply HRTF to the full sound
    binaural = hrtf.apply(idx, sound)  # returns float array [-1,1], shape (samples, 2)
    total_samples = len(binaural.data)
    
    # Only full snippets
    num_snippets = int(np.floor(total_samples / num_samples_per_snippet))
    
    for snippet_idx in range(num_snippets):
        start = snippet_idx * num_samples_per_snippet
        end = start + num_samples_per_snippet
        snippet_data = binaural.data[start:end].copy()
        
        # Fade-in/out to avoid clicks
        if snippet_data.shape[0] > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            snippet_data[:fade_samples, :] *= fade_in[:, None]
            snippet_data[-fade_samples:, :] *= fade_out[:, None]
        
        # Convert snippet to 16-bit PCM for WAV saving
        snippet_int16 = np.int16(np.clip(snippet_data, -1, 1) * 32767)
        
        # Build filename with overall min/max in 16-bit PCM
        filename = (
            f"az{int(round(azimuth))}_el{int(round(elevation))}_"
            f"snippet{snippet_idx:03d}_min{overall_min}_max{overall_max}.wav"
        )
        filepath = os.path.join(out_folder, filename)
        
        # Save WAV
        wavfile.write(filepath, sr, snippet_int16)
    
    print(f"Processed KEMAR position {idx + 1}/{num_sources}")

print("All full-size snippets processed and saved!")