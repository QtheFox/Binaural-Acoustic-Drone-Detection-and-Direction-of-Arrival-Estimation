import os
import time
import torch
from Step_5B_training5 import SoundDirectionCNN, SoundDirectionDataset
import matplotlib.pyplot as plt
modelname="best_model4B_real2.pth"

def extract_azimuth_from_path(path):
    parts = path.split(os.sep)
    for p in parts:
        if p.startswith("Az"):
            return int(p.split("_")[0].replace("Az", ""))
    return None


def run_single_inference():
    dataset = SoundDirectionDataset("dataset_debug")

    # -----------------------------
    # Load one sample (measure load time)
    # -----------------------------
    t0 = time.time()
    left,right,corr, true_label = dataset[0]#456
    sample_path = dataset.samples[0]["left"]
    true_az_from_folder = extract_azimuth_from_path(sample_path)

    # Add batch dim
    left  = left.unsqueeze(0)
    right = right.unsqueeze(0)
    corr = corr.unsqueeze(0)

    t1 = time.time()
    load_time = (t1 - t0) * 1000  # ms

    # -----------------------------
    # Load model
    # -----------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SoundDirectionCNN().to(device)

    if not os.path.exists(modelname):
        print("ERROR: "+modelname+" not found!")
        return

    model.load_state_dict(torch.load(modelname, map_location=device))
    model.eval()

    left  = left.to(device)
    right = right.to(device)
    corr = corr.to(device)

    torch.save(left, "left_.pt")
    torch.save(right, "right_.pt")
    torch.save(corr, "corr_.pt")
  

    # -----------------------------
    # Measure inference time
    # -----------------------------
    torch.cuda.synchronize() if device.type == "cuda" else None
    t2 = time.time()

    with torch.no_grad():
        outputs = model(left, right, corr)

    torch.cuda.synchronize() if device.type == "cuda" else None
    t3 = time.time()

    inference_time = (t3 - t2) * 1000  # ms
    total_time = (t3 - t0) * 1000  # ms

    # -----------------------------
    # Interpret results
    # -----------------------------
    _, pred_class = torch.max(outputs, 1)
    pred_class = pred_class.item()

    predicted_azimuth = (pred_class * 12) % 360
    true_az_class = (true_label * 12) % 360

    print("\n--- Inference Results ---")
    print("True azimuth from folder:      ", true_az_from_folder, "°")
    print("True class index:              ", true_label)
    print("True azimuth (label ×12°):     ", true_az_class, "°")
    print("Predicted class index:         ", pred_class)
    print("Predicted azimuth (×12°):      ", predicted_azimuth, "°")

    print("\n--- Timing ---")
    print(f"Image load + preprocessing: {load_time:.3f} ms")
    print(f"Model inference only:       {inference_time:.3f} ms")
    print(f"Total time:                 {total_time:.3f} ms")

    print("\nRaw logits:\n", outputs.cpu())
    print("------------------------------")


if __name__ == "__main__":
    run_single_inference()