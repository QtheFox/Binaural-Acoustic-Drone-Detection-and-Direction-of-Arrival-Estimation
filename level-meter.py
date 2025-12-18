import sounddevice as sd
import numpy as np
import os

samplerate = 44100
channels = 2
block_duration = 0.1
input_device = 3
sd.default.device = input_device

def rms(signal):
    return np.sqrt(np.mean(np.square(signal)))

def vu_meter(db, width=50):
    # db-Wert in Balkenlänge umrechnen
    # -50 dB = leer, 0 dB = voll
    db = max(-50, min(0, db))   # begrenzen
    filled = int((db + 50) / 50 * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"

print("Starte VU-Meter... (Strg+C zum Beenden)\n")

try:
    with sd.InputStream(samplerate=samplerate,
                        channels=channels,
                        dtype='float32') as stream:

        while True:
            block, _ = stream.read(int(block_duration * samplerate))
            rms_left = rms(block[:, 0])
            rms_right = rms(block[:, 1])

            db_left = 20 * np.log10(rms_left + 1e-8)
            db_right = 20 * np.log10(rms_right + 1e-8)

            bar_left = vu_meter(db_left)
            bar_right = vu_meter(db_right)

            # Konsole „ersetzt“ die Zeilen
            os.system('cls' if os.name == 'nt' else 'clear')
            print("LIVE VU-METER")
            print(f"L {db_left:6.1f} dB {bar_left}")
            print(f"R {db_right:6.1f} dB {bar_right}")

except KeyboardInterrupt:
    print("\nBeendet.")