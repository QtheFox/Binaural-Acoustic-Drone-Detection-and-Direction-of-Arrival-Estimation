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
# CNN with binaural diff-filter init
# -----------------------------
class BinauralCNN(nn.Module):
    def __init__(self, output_features=128, diff_fraction=0.5):
        super().__init__()

        # Input channels: left, right, correlogram
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self._init_diff_filters(self.conv1, diff_fraction)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveMaxPool2d((8, 8))
        self.fc = nn.Linear(64 * 8 * 8, output_features)

    def _init_diff_filters(self, conv_layer, diff_fraction):
        """
        Initialize a subset of filters to respond to LEFT - RIGHT.
        Channel mapping:
          0 = left
          1 = right
          2 = correlogram (kept small / neutral)
        """
        num_filters = conv_layer.weight.shape[0]
        num_diff = int(num_filters * diff_fraction)

        with torch.no_grad():
            for i in range(num_diff):
                w = torch.randn(conv_layer.weight.shape[2:]) * 0.1

                conv_layer.weight[i, 0, :, :] =  w    # left
                conv_layer.weight[i, 1, :, :] = -w    # right
                conv_layer.weight[i, 2, :, :] = 0.01 * w  # correlogram (weak)

                if conv_layer.bias is not None:
                    conv_layer.bias[i] = 0.0

    def forward(self, left, right, correlogram):
        x = torch.cat([left, right, correlogram], dim=1)

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
# Full sound direction model
# -----------------------------
class SoundDirectionCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = BinauralCNN(output_features=128)

        self.fc1 = nn.Linear(128, 128)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, 30)  # 30 azimuth classes

    def forward(self, left, right, correlogram):
        x = self.feature_extractor(left, right, correlogram)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.out(x)
        return x


# -----------------------------
# Dataset
# -----------------------------
class SoundDirectionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform

        az_folders = sorted(glob.glob(os.path.join(root_dir, 'Az*_El*')))
        for az_folder in az_folders:
            snippet_folders = sorted(glob.glob(os.path.join(az_folder, '*')))
            for snippet in snippet_folders:
                left = os.path.join(snippet, 'left.png')
                right = os.path.join(snippet, 'right.png')
                corr = os.path.join(snippet, 'correlogram.png')

                if all(os.path.exists(p) for p in [left, right, corr]):
                    az_str = os.path.basename(az_folder).split('_')[0]
                    az = int(az_str.replace('Az', ''))

                    self.samples.append({
                        'left': left,
                        'right': right,
                        'corr': corr,
                        'label': az
                    })

        self.num_classes = 30

        self.norm_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        left = Image.open(s['left']).convert('L')
        right = Image.open(s['right']).convert('L')
        corr = Image.open(s['corr']).convert('L')

        if self.transform:
            left = self.transform(left)
            right = self.transform(right)
            corr = self.transform(corr)
        else:
            left = self.norm_transform(left)
            right = self.norm_transform(right)
            corr = self.norm_transform(corr)

        label_idx = int(s['label'] / 12)
        return left, right, corr, label_idx


# -----------------------------
# Training loop
# -----------------------------
def main():
    dataset_training = 'dataset_training_real2'
    dataset_validation = 'dataset_validation_real2'

    train_dataset = SoundDirectionDataset(dataset_training)
    val_dataset = SoundDirectionDataset(dataset_validation)

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SoundDirectionCNN().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 100
    best_val_loss = float('inf')

    for epoch in range(epochs):
        # ---- Training ----
        model.train()
        running_loss, correct, total = 0, 0, 0

        for left, right, corr, labels in train_loader:
            left = left.to(device)
            right = right.to(device)
            corr = corr.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(left, right, corr)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total

        # ---- Validation ----
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0

        with torch.no_grad():
            for left, right, corr, labels in val_loader:
                left = left.to(device)
                right = right.to(device)
                corr = corr.to(device)
                labels = labels.to(device)

                outputs = model(left, right, corr)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model4B_real2.pth')
            print("✅ Saved best model")

    print("Training complete!")


if __name__ == '__main__':
    main()
