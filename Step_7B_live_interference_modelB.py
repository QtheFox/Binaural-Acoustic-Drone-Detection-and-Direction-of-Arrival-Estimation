#record sound from step 1B
#preprocessing from step 2 and step 3 but maybe more compressed
#interference from step 6
#display from step 7
#irgendwas funktioniert noch nicht
#in az84_el0_snippet 10 ordner sind die bilder mit denen von step_7 generierten ausgetauscht und step_6_interference leiferte
#im letzten test das richrtige ergebnis während step_7 ein anderers lieferte. das muss nochmal genauer überprüft
#werden ob das jetzt wirklich dieselben bilder sind und was im code nicht stimmt oder ob ich doch andere bilder hergebnommen habe

import sounddevice as sd
import numpy as np

from numba import njit, prange
from scipy.signal import butter, lfilter, find_peaks, correlate
import matplotlib.pyplot as plt
from PIL import Image
import time
from Step_5_training_modelB import SoundDirectionCNN
from scipy.io import wavfile
#load model
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SoundDirectionCNN()
model.load_state_dict(torch.load("best_model4B_real2.pth", map_location=device))
model.to(device)
model.eval()

# EXACT same normalization as training
norm = transforms.Normalize(mean=[0.5], std=[0.5])
max_delay_ms = 1.4

#tensor helper
def np_to_tensor(img_np):
    """
    img_np: (H, W) float or int
    returns: (1, 1, H, W) torch tensor
    """
    img = img_np.astype(np.float32)

    # scale to [0,1] if needed
    if img.max() > 1.0 or img.min() < 0.0:
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)

    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    t = norm(t)
    return t

def np_to_tensor2(img_np):
    """
    img_np: (H, W) float or int
    returns: (1, 1, H, W) torch tensor normalized like training
    """
    img = img_np.astype(np.float32)

    # if values are 0..255, scale to 0..1
    if img.max() > 1.0:
        img = img / 255.0

    # convert to tensor and add channel/batch dimensions
    t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # normalize exactly like transforms.Normalize(mean=[0.5], std=[0.5])
    t = (t - 0.5) / 0.5  # scales 0..1 → -1..1

    return t
# ============================================================
# TIMING HELPER
# ============================================================
def stamp(msg, t0):
    t = time.perf_counter()
    print(f"[{t - t0:8.3f} s] {msg}")
    return t

samplerate = 44100
channels = 2
record_duration = 0.1
input_device = 3  # M-Track Duo

fade_ms = 5             # optional fade-in/out
fade_samples = int(fade_ms / 1000 * samplerate)

#processing parameters
order = 2
bandwidth = 40
freqs = np.arange(800, 5001, 5)
threshold = 0
fs_analysis = 340 / 0.01  # 17,000 Hz for 2cm resolution


# -----------------------------
# Analysis function
# -----------------------------
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

def analyze_channel(signal, orig_fs):
    #send signal through a lot of narrow butterworth filters covering the whole frequency range and retrieve max amplitudes and 
    #corresponding time stamps
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
        y = lfilter(b, a, signal)

        peaks, _ = find_peaks(y)
        all_max_times.append(t[peaks])
        all_max_amplitudes.append(y[peaks])

    max_time = max([max(row) if len(row) > 0 else 0 for row in all_max_times])
    num_samples = int(np.ceil(max_time * fs_analysis)) + 1

    return fill_amplitudes_numba(all_max_times, all_max_amplitudes, fs_analysis, num_samples)


def analyze_channel_fast(signal, orig_fs):
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
    #arr = (arr * 256).astype(np.uint8)
    # Flip vertically to match origin='lower'
    arr = np.flipud(arr)
    return Image.fromarray(arr)


def simulate_saving(arr):

    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
         ])
    t = norm_transform(arr)       # shape (1, H, W)
    t = t.unsqueeze(0)               
    return t
#####################warm-up####################
# Warm-up call with small dummy arrays
t0 = time.perf_counter()
dummy_times = [np.array([0.0])]
dummy_amps  = [np.array([0.0])]
dummy_num_samples = 10
dummy_fs = 1.0

_ = fill_amplitudes_numba(dummy_times, dummy_amps, dummy_fs, dummy_num_samples)
t0 = stamp("Finished Warm up", t0)
#####################recording####################
sd.default.device = input_device
stream = sd.InputStream(samplerate=samplerate,
                        channels=channels,
                        dtype='float32')
stream.start()

running = True
while running:
    audio = sd.rec(int(record_duration * samplerate),
                        samplerate=samplerate,
                        channels=channels,
                        dtype='float32')
    sd.wait()
    #test###########
    #audio = wavfile.read("az96_el0_snippet006.wav")
    #audio=audio[1]
    ###############
    # t0 = stamp("Recorded Audio", t0)
    # max_first_20 = np.max(audio[:20, :])
    # max_last_20  = np.max(audio[-20:, :])
    # fade_in = np.linspace(0, 1, fade_samples)
    # fade_out = np.linspace(1, 0, fade_samples)
    # audio[:fade_samples, :] *= fade_in[:, None]
    # audio[-fade_samples:, :] *= fade_out[:, None]

    # Convert snippet to 16-bit PCM
    #leave away for test
    audio = np.int16(np.clip(audio, -1, 1) * 32767)
    # Save WAV
    #wavfile.write("test-90.wav", samplerate, audio)
    left = audio[:, 0].astype(np.float64)# / 32768.0
    right = audio[:, 1].astype(np.float64)# / 32768.0

    amp_left,fb_l = analyze_channel_with_fb(left, samplerate)
    amp_right,fb_r = analyze_channel_with_fb(right, samplerate)

    t0 = stamp("Cropped BW images", t0)
    #bis hier hin sind step_7 und step_6 gleich
    # ---- Resize continuous images (PIL)

    corr = compute_correlogram(fb_l, fb_r, fs_analysis, max_delay_ms)

    # ---- convert to images ----
    img_left = to_img(amp_left, -2000, 2000).resize((340, 84))
    img_right = to_img(amp_right, -2000, 2000).resize((340, 84))
    img_corr = to_img(corr, -1.0, 1.0).resize((340, 84))
    #prepare interference:
    # Create the transform
    left = np.array(img_left, dtype=np.float32)
    right = np.array(img_right, dtype=np.float32)
    corr = np.array(img_corr, dtype=np.float32)
    #check how image is saved, which values do the pixels have, compare with values that are loaded in step_6_interference, these have to
    #be the same otherwise no wonder it is not working
    # Apply to NumPy array (H, W)
    #################################
    #simulate saving and loading image as in step 6
    left_t   = simulate_saving(left)
    right_t  = simulate_saving(right)
    corr_t  = simulate_saving(corr)
    ###################################

    #left_long_t = to_tensor(amp_left_long)  
    #left_long_t   = np_to_tensor2(amp_left_long).to(device)
    #right_long_t  = np_to_tensor2(amp_right_long).to(device)
    #left_short_t  = np_to_tensor2(amp_left_bw_short).to(device)
    #right_short_t = np_to_tensor2(amp_right_bw_short).to(device)
    
    #left_t = torch.flip(left_t, dims=[-2]).to(device)
    #right_t = torch.flip(right_t, dims=[-2]).to(device)
    #corr_t = torch.flip(corr_t, dims=[-2]).to(device)
  

    torch.save(left_t, "left.pt")
    torch.save(right_t, "right.pt")
    torch.save(corr_t, "corr.pt")


    #interference
    with torch.no_grad():
        output = model(left_t,right_t,corr_t)

        #probs = torch.softmax(logits, dim=1)
        #pred_class = probs.argmax(dim=1).item()
        _, pred_class = torch.max(output, 1)
        pred_class = pred_class.item()
    pred_azimuth_deg = pred_class * 12

    print(f"Predicted azimuth class: {pred_class}")
    print(f"Predicted azimuth angle: {pred_azimuth_deg}°")
    #confidence = probs[0, pred_class].item()
    #print(f"Confidence: {confidence:.2f}")
    # def array_to_pil(img, vmin, vmax):
    #     # normalize to 0–255
    #     img = np.clip(img, vmin, vmax)
    #     img = (img - vmin) / (vmax - vmin)
    #     img = 1.0 - img 
    #     img = (img * 255).astype(np.uint8)
    #     img = np.flipud(img)
    #     return Image.fromarray(img, mode="L") 

    # img_left = array_to_pil(amp_left, vmin, vmax)
    # img_right = array_to_pil(amp_right, vmin, vmax)

    # img_left_bw = array_to_pil(amp_leftbw, vmin_bw, vmax_bw)
    # img_right_bw = array_to_pil(amp_rightbw, vmin_bw, vmax_bw)

    # width, height = img_left_bw.size
    # img_left_short = img_left_bw.crop((1650, 0, 1750, height))
    # img_right_short = img_right_bw.crop((1650, 0, 1750, height))
    # img_left_long = img_left.resize((340, 84), Image.BILINEAR)
    # img_right_long = img_right.resize((340, 84), Image.BILINEAR)

    # img_left_short.save("left_short.png")
    # img_right_short.save("right_short.png")
    # img_left_long.save("left_long.png")
    # img_right_long.save("right_long.png")
    # print("hi")