import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define EfficientNet-Lite-based Model
class EfficientNetLiteCustom(nn.Module):
    def __init__(self, num_classes=31):
        super(EfficientNetLiteCustom, self).__init__()
        # Load pretrained EfficientNet-B0 model
        self.base_model = models.efficientnet_b0(pretrained=True)

        # Modify the first layer to accept single-channel input (grayscale)
        self.base_model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)

        # Replace the classifier with a custom classifier for 31 classes
        self.base_model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.base_model.classifier[1].in_features, num_classes)
        )

    def forward(self, x):
        return self.base_model(x)

import os
import numpy as np
from sklearn.model_selection import train_test_split

# Define dataset path
dataset_path = os.path.expanduser('~') + '/custom/dataset/augmentation/'
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")

# Create directories for splits
os.makedirs(train_path, exist_ok=True)
os.makedirs(val_path, exist_ok=True)
os.makedirs(test_path, exist_ok=True)

# Define labels
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
          'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'greaterThan', 'comma', 
          'apostrophe', 'tilde', 'questionMark']

# Define test and validation split sizes
test_size = 0.2
val_size = 0.1

# Iterate through each character label
for i, label in enumerate(labels):
    print(f"Processing: {label}")

    # Load the character's dataset
    X = np.load(os.path.join(dataset_path, f"{label}.npy"))
    y = np.full(X.shape[0], fill_value=i)  # Assign class label

    # Split into train & test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=45)

    # Further split train into train & validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=45)

    # Apply reshaping to each split
    def reshape_and_save(X, path, name):
        # Add first reshaping: Expand along the second axis
        i, j, k = X.shape[0], 23, 192
        Matrix = np.array([[[-1 for _ in range(k)] for _ in range(j)] for _ in range(i)])
        X = np.concatenate((X, Matrix), axis=1)
        print(f"After first reshape {name}: {X.shape}")

        # Add second reshaping: Expand along the third axis
        i, j, k = X.shape[0], 224, 32
        Matrix = np.array([[[-1 for _ in range(k)] for _ in range(j)] for _ in range(i)])
        X = np.concatenate((X, Matrix), axis=2)
        print(f"After second reshape {name}: {X.shape}")

        # Save the reshaped data
        np.save(os.path.join(path, f"{label}.npy"), X)

    # Apply reshaping and saving
    reshape_and_save(X_train, train_path, "X_train")
    reshape_and_save(X_val, val_path, "X_val")
    reshape_and_save(X_test, test_path, "X_test")

    # Save labels
    np.save(os.path.join(train_path, f"{label}_labels.npy"), y_train)
    np.save(os.path.join(val_path, f"{label}_labels.npy"), y_val)
    np.save(os.path.join(test_path, f"{label}_labels.npy"), y_test)

print("Train, validation, and test datasets saved successfully per character with reshaping!")

# Function to load all data from a directory
def load_split_data(split_path, labels):
    X_list, y_list = [], []
    
    for i, label in enumerate(labels):
        x_data = np.load(os.path.join(split_path, f"{label}.npy"))
        y_data = np.load(os.path.join(split_path, f"{label}_labels.npy"))
        
        X_list.append(x_data)
        y_list.append(y_data)

    X = np.vstack(X_list)  # Stack all character data together
    y = np.concatenate(y_list)  # Stack labels

    return X, y

# Load train, validation, and test datasets
X_train, y_train = load_split_data(train_path, labels)
X_val, y_val = load_split_data(val_path, labels)
X_test, y_test = load_split_data(test_path, labels)

# Convert data to float and labels to integers
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('int64')
y_val = y_val.astype('int64')
y_test = y_test.astype('int64')

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)  # Single-channel input
X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)

y_train = torch.tensor(y_train).to(device)
y_val = torch.tensor(y_val).to(device)
y_test = torch.tensor(y_test).to(device)

print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {y_val.shape}")
print(f"Test Data Shape: {X_test.shape}, {y_test.shape}")
# Define model, loss function, and optimizer
num_classes = 31
model = EfficientNetLiteCustom(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 16
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
# Function to calculate accuracy
def calculate_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total * 100

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

    # Calculate accuracy for training and validation sets
    train_accuracy = calculate_accuracy(model, train_loader)
    val_accuracy = calculate_accuracy(model, val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

# Save the model
torch.save(model.state_dict(), "efficientnet_lite_model_only_on_training.pth")
print("Model saved successfully.")

# Evaluate the model
model.eval()
test_dataset = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=1)
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
