# signal amplitude adjustment in sim path: 
# first in synth_data_gen2 signal is more or less between -1 and 1 then is converted to +- 32767, 3.6 seconds dronesound split into 0.1 seconds snippets
#for handover the filename is important. Azimuth, Elevation, Snippet Index
# in Step_2_preprocess_synth_el0 signal is translated into 2 spectrums, one spectrum is cut between -2000 and 2000
# the other spectrum between 0 and 1
# in Step_3_preprocess_synth_el0_2 data is reorganized into dataset folder structure and small part of bw image gets cropped
# in Step_4_split_dataset dataset is split into training and validation data with a ratio of 80 20
# first in synth_data_gen2 signal is more or less between -1 and 1 then is converted to +- 32767, 3.6 seconds dronesound split into 0.1 seconds snippets
#for handover the filename is important. Azimuth, Elevation, Snippet Index
#record 3.6 seconds, split into 36 snippets and add fade
import pygame
import math
import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
import time
import os
from scipy.io import wavfile

# ------------------ EINSTELLUNGEN ------------------
WIDTH, HEIGHT = 500, 500
CENTER = (WIDTH // 2, HEIGHT // 2)
RADIUS = 200
ANGLE_STEP = 12
out_folder = "dronesound_hrtf_snippets_real"
os.makedirs(out_folder, exist_ok=True)
samplerate = 44100
snippet_duration=0.1
fade_ms = 5             # optional fade-in/out
num_samples_per_snippet = int(snippet_duration * samplerate)
fade_samples = int(fade_ms / 1000 * samplerate)
channels = 2
threshold_db = -25
record_duration = 3.6
input_device = 3  # M-Track Duo

sd.default.device = input_device

# ----------------------------------------------------

def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def get_db_left_right(block):
    rms_left = rms(block[:, 0])
    rms_right = rms(block[:, 1])
    db_left = 20 * np.log10(rms_left + 1e-8)
    db_right = 20 * np.log10(rms_right + 1e-8)
    return db_left, db_right


# ------------------ PYGAME SETUP ------------------
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ziffernblatt mit Audio-Trigger")
clock = pygame.time.Clock()
font_big = pygame.font.SysFont(None, 48)
font_small = pygame.font.SysFont(None, 32)

azimuth = 0
trigger_ready = True

print("Starte Audio-Überwachung und Pygame...")

# ------------------ AUDIO-STREAM ------------------
stream = sd.InputStream(samplerate=samplerate,
                        channels=channels,
                        dtype='float32')
stream.start()


running = True
while running:

    # ------------------ EVENTS ------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # ------------------ AUDIO AUSLESEN ------------------
    block, _ = stream.read(int(0.1 * samplerate))
    db_l, db_r = get_db_left_right(block)

    loud = (db_l > threshold_db) or (db_r > threshold_db)

    # ------------------ TRIGGER LOGIK ------------------
    if loud and trigger_ready:

        print(f"Trigger! Lautstärke: L={db_l:.1f} dB, R={db_r:.1f} dB")
        print("→ Starte 1 Sekunde Aufnahme...")

        audio = sd.rec(int(record_duration * samplerate),
                       samplerate=samplerate,
                       channels=channels,
                       dtype='float32')
        sd.wait()
        #az0_el-10_snippet002_min-3378_max3158
        #create 36 0.1 sec snippets
        total_samples = len(audio)
    
        # Only full snippets
        num_snippets = int(np.floor(total_samples / num_samples_per_snippet))
        for snippet_idx in range(num_snippets):
            start = snippet_idx * num_samples_per_snippet
            end = start + num_samples_per_snippet
            snippet_data = audio[start:end].copy()
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
                    f"az{int(round(azimuth))}_el0_"
                    f"snippet{snippet_idx:03d}.wav"
                )
                filepath = os.path.join(out_folder, filename)
                
                # Save WAV
                wavfile.write(filepath, samplerate, snippet_int16)


        

        #fade in - fade out
        filename = f"az{azimuth}_el0_snippet.wav"
        write(filename, samplerate, audio)
        print(f"✔ gespeichert: {filename}")

        azimuth = (azimuth + ANGLE_STEP) % 360
        trigger_ready = False

    if not loud:
        trigger_ready = True

    # ------------------ PYGAME ZEICHNEN ------------------
    screen.fill((255, 255, 255))

    # Kreis
    pygame.draw.circle(screen, (0, 0, 0), CENTER, RADIUS, 3)

    # Zeiger
    rad = math.radians(azimuth - 90)
    x = CENTER[0] + RADIUS * 0.9 * math.cos(rad)
    y = CENTER[1] + RADIUS * 0.9 * math.sin(rad)
    pygame.draw.line(screen, (255, 0, 0), CENTER, (x, y), 5)

    # Winkel-Anzeige (mittel)
    text_angle = font_big.render(f"{azimuth}°", True, (0, 0, 0))
    screen.blit(text_angle, text_angle.get_rect(center=(CENTER[0], CENTER[1] - 20)))

    # Lautstärke-Anzeige (klein, darunter)
    text_volume = font_small.render(
        f"L: {db_l:.1f} dB   R: {db_r:.1f} dB",
        True,
        (0, 0, 0)
    )
    screen.blit(text_volume, text_volume.get_rect(center=(CENTER[0], CENTER[1] + 30)))

    pygame.display.flip()
    clock.tick(30)

pygame.quit()
stream.stop()
stream.close()
print("Beendet.")