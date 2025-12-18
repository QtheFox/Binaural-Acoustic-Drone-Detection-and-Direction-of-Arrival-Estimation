import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F

# -----------------------------
# Binaural CNN branch with difference channel
# -----------------------------
class BinauralCNNBranch(nn.Module):
    def __init__(self, input_channels=1, output_features=128, diff_fraction=0.5):
        super().__init__()
        # 3 channels: left, right, left-right difference
        self.conv1 = nn.Conv2d(input_channels*3, 16, 3, padding=1)
        self._init_diff_filters(self.conv1, diff_fraction)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((8,8))
        self.fc = nn.Linear(64*8*8, output_features)

    def _init_diff_filters(self, conv_layer, diff_fraction):
        """Initialize some filters to respond to left-right difference."""
        num_filters = conv_layer.weight.shape[0]
        num_diff = int(num_filters * diff_fraction)
        with torch.no_grad():
            for i in range(num_diff):
                w = torch.randn(conv_layer.weight.shape[2:]) * 0.1
                conv_layer.weight[i,0,:,:] = w        # left
                conv_layer.weight[i,1,:,:] = -w       # right
                conv_layer.weight[i,2,:,:] = 0.01*w   # difference small init

    def forward(self, left, right):
        diff = left - right
        x = torch.cat([left, right, diff], dim=1)
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = F.relu(self.conv3(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        return x

# -----------------------------
# Full model with two branches
# -----------------------------
class SoundDirectionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.long_branch = BinauralCNNBranch()
        self.short_branch = BinauralCNNBranch()
        self.fc1 = nn.Linear(128*2, 256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.out = nn.Linear(128, 30)  # 30 direction classes

    def forward(self, left_long, right_long, left_short, right_short):
        long_feat = self.long_branch(left_long, right_long)
        short_feat = self.short_branch(left_short, right_short)
        x = torch.cat([long_feat, short_feat], dim=1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.out(x)
        return x

# -----------------------------
# Dataset with normalization
# -----------------------------
class SoundDirectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        az_folders = sorted(glob.glob(os.path.join(root_dir, 'Az*_El*')))
        for az_folder in az_folders:
            snippet_folders = sorted(glob.glob(os.path.join(az_folder, '*')))
            for snippet in snippet_folders:
                left_long = os.path.join(snippet, 'left_long.png')
                left_short = os.path.join(snippet, 'left_short.png')
                right_long = os.path.join(snippet, 'right_long.png')
                right_short = os.path.join(snippet, 'right_short.png')
                if all(os.path.exists(p) for p in [left_long, left_short, right_long, right_short]):
                    az_str = os.path.basename(az_folder).split('_')[0]
                    az = int(az_str.replace('Az',''))
                    self.samples.append({
                        'left_long': left_long,
                        'left_short': left_short,
                        'right_long': right_long,
                        'right_short': right_short,
                        'label': az
                    })
        self.num_classes = 30

        # Normalization: mean=0.5, std=0.5 (for grayscale)
        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        imgs = {}
        for key in ['left_long','left_short','right_long','right_short']:
            img = Image.open(s[key]).convert('L')
            if self.transform:
                img = self.transform(img)
            else:
                img = self.norm_transform(img)
            imgs[key] = img
        #label_idx = round(s['label'] / 12) % 30
        label_idx = int(s['label'] / 12)
        return imgs['left_long'], imgs['right_long'], imgs['left_short'], imgs['right_short'], label_idx

# -----------------------------
# Training loop
# -----------------------------
def main():
    #dataset_training='dataset_training'
    #dataset_validation='dataset_validation'
    dataset_training='dataset_training_real'
    dataset_validation='dataset_validation_real'
    train_dataset = SoundDirectionDataset(dataset_training)
    val_dataset = SoundDirectionDataset(dataset_validation)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SoundDirectionCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # --- Training ---
        model.train()
        running_loss, correct, total = 0, 0, 0
        for le, re, lt, rt, labels in train_loader:
            le, re, lt, rt, labels = le.to(device), re.to(device), lt.to(device), rt.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(le, re, lt, rt)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # --- Validation ---
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for le, re, lt, rt, labels in val_loader:
                le, re, lt, rt, labels = le.to(device), re.to(device), lt.to(device), rt.to(device), labels.to(device)
                outputs = model(le, re, lt, rt)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(f"Epoch {epoch+1}/{epochs}: Train Loss {train_loss:.4f}, Acc {train_acc:.4f} | Val Loss {val_loss:.4f}, Acc {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model4.pth')
            print("Saved best model.")

    print("Training complete!")

if __name__ == '__main__':
    main()