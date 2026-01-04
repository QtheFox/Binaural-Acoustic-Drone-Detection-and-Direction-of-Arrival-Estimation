import os
from PIL import Image
import matplotlib.pyplot as plt
#input_folder = "plots_left_right_snippets_el0_bin"
#output_folder = "dataset"
#input_folder = "plots_left_right_snippets_el0_bin_real"
input_folder = "dronesound_debug"
output_folder = "dronesound_debug"
# Create folders if they don't exist
def make_dirs(path):
    os.makedirs(path, exist_ok=True)

# Collect all files
files = os.listdir(input_folder)

# Group files by snippet and side
snippets = {}
for filename in files:
    if not filename.endswith(".png"):
        continue
    name = filename.replace(".png", "")
    parts = name.split("_")
    az = parts[0][2:]        # remove 'az'
    el = parts[1][2:]        # remove 'el'
    snippet = parts[2][7:]    # remove 'snippet'
    side = parts[3]            # left or right
    bw = "_bw" in name         # True if it's bw file

    key = (az, el, snippet, side)
    if key not in snippets:
        snippets[key] = {}
    if bw:
        snippets[key]["short"] = filename
    else:
        snippets[key]["long"] = filename

# Process each snippet
for key, file_dict in snippets.items():
    az, el, snippet, side = key
    snippet_folder = os.path.join(output_folder, f"Az{az}_El{el}", snippet)
    make_dirs(snippet_folder)

    # Process long image
    if "long" in file_dict:
        img_path = os.path.join(input_folder, file_dict["long"])
        img = Image.open(img_path)
        img_long = img.resize((340, 84))
        img_long.save(os.path.join(snippet_folder, f"{side}_long.png"))

    # Process short image (bw)
    if "short" in file_dict:
        img_path = os.path.join(input_folder, file_dict["short"])
        img = Image.open(img_path)
        width, height = img.size
        if width >= 1750:
            img_short = img.crop((1650, 0, 1750, height))
        else:
            img_short = img.crop((0, 0, width, height))  # fallback if too small
        img_short.save(os.path.join(snippet_folder, f"{side}_short.png"))

print("Dataset organized and resized successfully!")