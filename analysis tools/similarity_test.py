import cv2
from skimage.metrics import structural_similarity as ssim

# Load images
img1 = cv2.imread("out_right_az84_el0_snippet010.png")
img2 = cv2.imread("right_long.png")

# Convert to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Resize images to the same size (important!)
gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))

# Calculate SSIM
score, diff = ssim(gray1, gray2, full=True)

print(f"Similarity score: {score}")