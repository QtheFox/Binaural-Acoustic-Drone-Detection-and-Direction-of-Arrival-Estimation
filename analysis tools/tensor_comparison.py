import torch
import matplotlib.pyplot as plt

# Pairs of tensors to compare: (original, real-time)
tensor_files = [
    ("left_.pt", "left.pt", "Left"),
    ("right_.pt", "right.pt", "Right"),
    ("corr_.pt", "corr.pt", "corr"),
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


# import torch
# import matplotlib.pyplot as plt

# # Tensors to plot: (filename, title)
# tensor_files = [
#     ("left.pt", "Left"),
#     ("right.pt", "Right"),
#     ("corr.pt", "Correlation")
# ]

# fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# for ax, (fname, title) in zip(axes, tensor_files):
#     # Load tensor on CPU
#     tensor = torch.load(fname, map_location="cpu").squeeze().numpy()

#     # Plot
#     im = ax.imshow(tensor, cmap="viridis", origin="lower")
#     ax.set_title(title)
#     ax.axis("off")

#     # Colorbar
#     plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# plt.tight_layout()
# plt.show()
