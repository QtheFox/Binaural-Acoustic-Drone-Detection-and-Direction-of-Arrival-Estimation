import sounddevice as sd
import numpy as np
samplerate = 44100
channels = 2
record_duration = 3.6
input_device = 3  # M-Track Duo
from scipy.io import wavfile
sd.default.device = input_device
stream = sd.InputStream(samplerate=samplerate,
                        channels=channels,
                        dtype='float32')
stream.start()
audio = sd.rec(int(record_duration * samplerate),
                       samplerate=samplerate,
                       channels=channels,
                       dtype='float32') 
audio= np.int16(np.clip(audio, -1, 1) * 32767)

wavfile.write("test.wav", samplerate, audio)