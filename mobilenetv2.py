import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define dataset paths
dataset_path = os.path.expanduser('~') + '/custom/dataset/augmentation/'
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")

# Define labels
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
          'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'greaterThan', 'comma',
          'apostrophe', 'tilde', 'questionMark']

# Function to load dataset from pre-split files
def load_split_data(split_path, labels):
    X_list, y_list = [], []
    
    for i, label in enumerate(labels):
        x_data = np.load(os.path.join(split_path, f"{label}.npy"))
        y_data = np.load(os.path.join(split_path, f"{label}_labels.npy"))

        X_list.append(x_data)
        y_list.append(y_data)

    X = np.vstack(X_list)  # Stack all data together
    y = np.concatenate(y_list)  # Stack labels

    return X, y

# Load pre-split train, validation, and test datasets
X_train, y_train = load_split_data(train_path, labels)
X_val, y_val = load_split_data(val_path, labels)
X_test, y_test = load_split_data(test_path, labels)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.astype('float32'), dtype=torch.float32).unsqueeze(1).to(device)  # Single-channel input
X_val = torch.tensor(X_val.astype('float32'), dtype=torch.float32).unsqueeze(1).to(device)
X_test = torch.tensor(X_test.astype('float32'), dtype=torch.float32).unsqueeze(1).to(device)

y_train = torch.tensor(y_train.astype('int64')).to(device)
y_val = torch.tensor(y_val.astype('int64')).to(device)
y_test = torch.tensor(y_test.astype('int64')).to(device)

print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {y_val.shape}")
print(f"Test Data Shape: {X_test.shape}, {y_test.shape}")

# Define MobileNetV2-based model
class MobileNetV2Custom(nn.Module):
    def __init__(self, num_classes=31):
        super(MobileNetV2Custom, self).__init__()
        # Load pre-trained MobileNetV2 and modify
        self.base_model = models.mobilenet_v2(pretrained=True)  # Set `weights='IMAGENET1K_V1'` if pretraining is needed
        self.base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)  # Adjust for single-channel input
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base_model.last_channel, num_classes)  # Output layer for 31 classes
        )

    def forward(self, x):
        return self.base_model(x)

# Initialize model, loss function, and optimizer
num_classes = len(labels)
model = MobileNetV2Custom(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 16

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Train the model
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    # Print training loss per epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Save the trained model
model_path = os.path.join(dataset_path, "mobilenetv2_model_only_on_training.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Evaluate the model on test set
model.eval()
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1)

correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Print final test accuracy
print(f"Test Accuracy: {100 * correct / total:.2f}%")
