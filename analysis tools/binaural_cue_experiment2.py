import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Load images ---
img1 = Image.open("dronesound_debug/az100_el0_snippet005_right_bw.png").convert("RGB")
img2 = Image.open("dronesound_debug/az100_el0_snippet005_left_bw.png").convert("RGB")

# Convert to numpy arrays
arr1 = np.array(img1, dtype=np.float32)
arr2 = np.array(img2, dtype=np.float32)

# --- Invert images ---
arr1_inv = 255 - arr1
arr2_inv = 255 - arr2

# --- Parameters ---
num_shifts = 20
pixel_sums = []

# --- Calculate shifted sums ---
for shift in range(1, num_shifts + 1):
    # Shift arr2_inv left by `shift` pixels
    shifted_arr2 = np.zeros_like(arr2_inv)
    if shift < arr2_inv.shape[1]:
        shifted_arr2[:, :-shift, :] = arr2_inv[:, shift:, :]
    
    # Add images
    added_shifted = arr1_inv + shifted_arr2
    added_shifted = np.clip(added_shifted, 0, 255)
    
    # Sum of all pixels
    total_sum = np.sum(added_shifted)
    pixel_sums.append(total_sum)

pixel_sums = np.array(pixel_sums)

# --- Plot as colors ---
plt.figure(figsize=(12, 2))
plt.scatter(range(1, num_shifts + 1), np.ones_like(pixel_sums), c=pixel_sums, cmap='viridis', s=200)
plt.colorbar(label='Pixel sum')
plt.xlabel('Pixel shift')
plt.yticks([])  # hide y-axis
plt.title('Pixel sum for each shifted addition')
plt.show()