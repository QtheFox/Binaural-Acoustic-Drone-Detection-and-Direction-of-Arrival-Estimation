import torch
import matplotlib.pyplot as plt

# Pairs of tensors to compare: (original, real-time)
tensor_files = [
    ("left_long.pt", "left_long_rt.pt", "Left Long"),
    ("right_long.pt", "right_long_rt.pt", "Right Long"),
    ("left_short.pt", "left_short_rt.pt", "Left Short"),
    ("right_short.pt", "right_short_rt.pt", "Right Short")
]

for orig_file, rt_file, title in tensor_files:
    # Load tensors on CPU
    orig = torch.load(orig_file, map_location="cpu").squeeze().numpy()
    rt   = torch.load(rt_file, map_location="cpu").squeeze().numpy()

    # Check shape
    if orig.shape != rt.shape:
        print(f"[WARNING] {title}: shapes differ {orig.shape} vs {rt.shape}")
        continue

    # Compute difference
    diff = orig - rt

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(orig, cmap='viridis', origin='lower')
    axes[0].set_title(f'{title} - Original')
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)  # colorbar for Original

    im1 = axes[1].imshow(rt, cmap='viridis', origin='lower')
    axes[1].set_title(f'{title} - Real-time')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)  # colorbar for Real-time

    im2 = axes[2].imshow(diff, cmap='bwr', origin='lower')
    axes[2].set_title(f'{title} - Difference')
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)  # colorbar for Difference
    
    for ax in axes:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print max difference
    print(f"{title}: max absolute difference = {abs(diff).max():.6f}")
