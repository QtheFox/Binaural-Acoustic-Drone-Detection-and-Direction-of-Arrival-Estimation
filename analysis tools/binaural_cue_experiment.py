import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# --- Load images ---
img1 = Image.open("dronesound_debug/az100_el0_snippet005_right_bw.png").convert("RGB")
img2 = Image.open("dronesound_debug/az100_el0_snippet005_left_bw.png").convert("RGB")

# Convert to numpy arrays
arr1 = np.array(img1, dtype=np.float32)
arr2 = np.array(img2, dtype=np.float32)

# Make sure images have the same shape
assert arr1.shape == arr2.shape, "Images must be the same size"

# --- Add images (no shift) ---
added = arr1 + arr2

# Clip values to valid image range
added = np.clip(added, 0, 255).astype(np.uint8)

# --- Shift image2 left by 3 pixels ---
shift = 12
shifted_arr2 = np.zeros_like(arr2)
shifted_arr2[:, :-shift, :] = arr2[:, shift:, :]

# --- Add images (with shift) ---
added_shifted = arr1 + shifted_arr2
added_shifted = np.clip(added_shifted, 0, 255).astype(np.uint8)

# --- Plot results ---
plt.figure(figsize=(12, 4))

plt.subplot(2, 1, 1)
plt.title("Added (no shift)")
plt.imshow(added)
plt.axis("off")

# plt.subplot(1, 3, 2)
# plt.title("Image 2 shifted left by 3 px")
# plt.imshow(shifted_arr2.astype(np.uint8))
# plt.axis("off")

plt.subplot(2, 1, 2)
plt.title("Added (with shift)")
plt.imshow(added_shifted)
plt.axis("off")

plt.tight_layout()
plt.show()