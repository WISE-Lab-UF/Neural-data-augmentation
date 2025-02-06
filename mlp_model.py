import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

# Flatten the input images for MLP (Convert 2D data to 1D)
X_train = X_train.reshape(X_train.shape[0], -1)
X_val = X_val.reshape(X_val.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train.astype('float32')).to(device)
X_val = torch.tensor(X_val.astype('float32')).to(device)
X_test = torch.tensor(X_test.astype('float32')).to(device)

y_train = torch.tensor(y_train.astype('int64')).to(device)
y_val = torch.tensor(y_val.astype('int64')).to(device)
y_test = torch.tensor(y_test.astype('int64')).to(device)

print(f"Training Data Shape: {X_train.shape}, {y_train.shape}")
print(f"Validation Data Shape: {X_val.shape}, {y_val.shape}")
print(f"Test Data Shape: {X_test.shape}, {y_test.shape}")

# Define a Simple MLP Model
class MLP(nn.Module):
    def __init__(self, input_size, num_classes=31):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x  # No activation (CrossEntropyLoss applies softmax)

# Initialize model, loss function, and optimizer
input_size = X_train.shape[1]  # Flattened input size
num_classes = len(labels)
model = MLP(input_size=input_size, num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

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
    correct_train, total_train = 0, 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        
        # Compute training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_accuracy = 100 * correct_train / total_train

    # Validation phase
    model.eval()
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_accuracy = 100 * correct_val / total_val

    # Print training and validation accuracy per epoch
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")

# Save the trained model
model_path = os.path.join(dataset_path, "mlp_model_only_on_training.pth")
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")

# Evaluate the model on the test set
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
