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
from scipy.signal import butter, lfilter, find_peaks
import matplotlib.pyplot as plt
from PIL import Image
import time
from Step_5_training4 import SoundDirectionCNN
from scipy.io import wavfile
#load model
import torch
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SoundDirectionCNN()
model.load_state_dict(torch.load("best_model4_real1.pth", map_location=device))
model.to(device)
model.eval()

# EXACT same normalization as training
norm = transforms.Normalize(mean=[0.5], std=[0.5])

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

def simulate_saving(arr):
    # Normalize to 0-1
    vmin=-2000
    vmax=2000
    normalized = (arr - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)  # clip values outside vmin/vmax
        # Normalization: mean=0.5, std=0.5 (for grayscale)
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
         ])
    normalized = (normalized * 255).astype(np.uint8)
    t = norm_transform(normalized)       # shape (1, H, W)
    t=-t
    t = t.unsqueeze(0)               
    # Scale to 0-255 and convert to uint8 (simulate grayscale PNG)
    return t
def simulate_saving2(arr):
    # Normalize to 0-1
    vmin=0
    vmax=1
    normalized = (arr - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0, 1)  # clip values outside vmin/vmax
        # Normalization: mean=0.5, std=0.5 (for grayscale)
    norm_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
         ])
    normalized = (normalized * 255).astype(np.uint8)
    t = norm_transform(normalized)       # shape (1, H, W)
    t=-t
    t = t.unsqueeze(0)               
    # Scale to 0-255 and convert to uint8 (simulate grayscale PNG)
    return t
#####################warm-up####################
# Warm-up call with small dummy arrays
t0 = time.perf_counter()
dummy_times = [np.array([0.0])]
dummy_amps  = [np.array([0.0])]
dummy_num_samples = 10
dummy_fs = 1.0

_ = fill_amplitudes_numba_fast(dummy_times, dummy_amps, dummy_fs, dummy_num_samples)
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
    # audio = wavfile.read("test.wav")
    # audio=audio[1]
    ###############
    t0 = stamp("Recorded Audio", t0)
    max_first_20 = np.max(audio[:20, :])
    max_last_20  = np.max(audio[-20:, :])
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    audio[:fade_samples, :] *= fade_in[:, None]
    audio[-fade_samples:, :] *= fade_out[:, None]

    # Convert snippet to 16-bit PCM
    #leave away for test
    audio = np.int16(np.clip(audio, -1, 1) * 32767)
    # Save WAV
    wavfile.write("test-90.wav", samplerate, audio)
    left = audio[:, 0].astype(np.float64)# / 32768.0
    right = audio[:, 1].astype(np.float64)# / 32768.0

    amp_left,amp_leftbw = analyze_channel_fast(left, samplerate)
    amp_right,amp_rightbw = analyze_channel_fast(right, samplerate)
    ###############################
    #debug
    amp_left_s2,amp_leftbw_s2 = analyze_channel(left, samplerate)
    ###############################
    amp_left_bw_short  = amp_leftbw[:, 1650:1750]
    amp_right_bw_short = amp_rightbw[:, 1650:1750]
    t0 = stamp("Cropped BW images", t0)
    #bis hier hin sind step_7 und step_6 gleich
    # ---- Resize continuous images (PIL)
    a=amp_left
    b=amp_right
    amp_left  = Image.fromarray(amp_left.astype(np.float32))
    amp_right = Image.fromarray(amp_right.astype(np.float32))

    amp_left_long  = amp_left.resize((340, 84), Image.BILINEAR)
    amp_right_long = amp_right.resize((340, 84), Image.BILINEAR)

    amp_left_long = np.array(amp_left_long, dtype=np.float32)
    amp_right_long = np.array(amp_right_long, dtype=np.float32)
    t0 = stamp("Resized amplitude images", t0)

    # ---- Save images
    plt.imsave("out_left_az84_el0_snippet010.png",  amp_left_long,  cmap="binary", origin="lower",
            vmin=-2000, vmax=2000)
    plt.imsave("out_right_az84_el0_snippet010.png", amp_right_long, cmap="binary", origin="lower",
            vmin=-2000, vmax=2000)

    plt.imsave("out_left_bw_az84_el0_snippet010.png",  amp_left_bw_short,  cmap="binary", origin="lower",
            vmin=0, vmax=1)
    plt.imsave("out_right_bw_az84_el0_snippet010.png", amp_right_bw_short, cmap="binary", origin="lower",
            vmin=0, vmax=1)
    t0 = stamp("Saved images", t0)
    #prepare interference:
    # Create the transform
    to_tensor = transforms.ToTensor()  # just converts HxW -> (C,H,W) and scales 0..255 to 0..1 if uint8
    #check how image is saved, which values do the pixels have, compare with values that are loaded in step_6_interference, these have to
    #be the same otherwise no wonder it is not working
    # Apply to NumPy array (H, W)
    #################################
    #simulate saving and loading image as in step 6
    left_long_t   = simulate_saving(amp_left_long)
    right_long_t  = simulate_saving(amp_right_long)
    left_short_t  = simulate_saving2(amp_left_bw_short)
    right_short_t = simulate_saving2(amp_right_bw_short)
    ###################################

    #left_long_t = to_tensor(amp_left_long)  
    #left_long_t   = np_to_tensor2(amp_left_long).to(device)
    #right_long_t  = np_to_tensor2(amp_right_long).to(device)
    #left_short_t  = np_to_tensor2(amp_left_bw_short).to(device)
    #right_short_t = np_to_tensor2(amp_right_bw_short).to(device)
    
    left_long_t = torch.flip(left_long_t, dims=[-2]).to(device)
    right_long_t = torch.flip(right_long_t, dims=[-2]).to(device)
    left_short_t = torch.flip(left_short_t, dims=[-2]).to(device)
    right_short_t = torch.flip(right_short_t, dims=[-2]).to(device)

    # torch.save(left_long_t, "left_long_rt.pt")
    # torch.save(right_long_t, "right_long_rt.pt")
    # torch.save(left_short_t, "left_short_rt.pt")
    # torch.save(right_short_t, "right_short_rt.pt")

    #interference
    with torch.no_grad():
        output = model(left_long_t,right_long_t,left_short_t,right_short_t)

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