import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define LeNet-5 Model in PyTorch
class LeNet5(nn.Module):
    def __init__(self, num_classes=31):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)  # Adjusted for 224x224 input
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)  # No softmax since CrossEntropyLoss includes it
        return x

# Define dataset path
dataset_path = os.path.expanduser('~') + '/custom/dataset/augmentation/'
train_path = os.path.join(dataset_path, "train")
val_path = os.path.join(dataset_path, "val")
test_path = os.path.join(dataset_path, "test")

# Define labels
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
          'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'greaterThan', 'comma',
          'apostrophe', 'tilde', 'questionMark']

def load_data(path, labels, val_size, test_size):

    # Just to show on console
    for i, label in enumerate(labels):
      print(i, ' ', label)

    # for the first class
    X = np.load(path + labels[0] + '.npy')
    y = np.full(X.shape[0], fill_value= 0)

    # Split all classes among Train, Validation & Test - SET
    for i, label in enumerate(labels[1:]):
        x = np.load(path + label + '.npy')

        X = np.vstack((X, x))
        y = np.append(y, np.full(x.shape[0], fill_value=i+1))


    return (X, y)

labels=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','greaterThan','comma','apostrophe','tilde','questionMark']
X, y = load_data(os.path.expanduser('~') + '/custom/dataset/', labels=labels, val_size=0, test_size=0)

# zeros=np.zeros((3627,23,192))
# X=np.concatenate((X,zeros),axis=1)
# print(X.shape, y.shape)

# zeros2=np.zeros((3627,224,32))
# X=np.concatenate((X,zeros2),axis=2)
# print(X.shape, y.shape)

i, j, k = X.shape[0], 23, 192
Matrix = np.array([[[-1 for x in range(k)] for y in range(j)] for z in range(i)]) 
print(Matrix.shape)
X=np.concatenate((X,Matrix),axis=1)
print(X.shape, y.shape)
i, j, k = X.shape[0], 224, 32
Matrix = np.array([[[-1 for x in range(k)] for y in range(j)] for z in range(i)]) 
X=np.concatenate((X,Matrix),axis=2)
print(X.shape, y.shape)

test_size = 0.2
val_size = 0.1

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=45)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=45)


print(X_train.shape, ' ', y_train.shape)
print(X_val.shape, ' ', y_val.shape)
print(X_test.shape, ' ', y_test.shape)

print('\n')



num_classes = len(labels)
# Convert data to float32
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')

y_train = y_train.astype('int64')
y_val = y_val.astype('int64')
y_test = y_test.astype('int64')

# Convert to PyTorch tensors
X_train = torch.tensor(X_train).unsqueeze(1).to(device)  # Add channel dimension
X_val = torch.tensor(X_val).unsqueeze(1).to(device)
X_test = torch.tensor(X_test).unsqueeze(1).to(device)

y_train = torch.tensor(y_train).to(device)
y_val = torch.tensor(y_val).to(device)
y_test = torch.tensor(y_test).to(device)

# Define model, loss function, and optimizer
num_classes = len(labels)
model = LeNet5(num_classes=num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
batch_size = 16

# Create DataLoaders
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_dataset = TensorDataset(X_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Track loss and accuracy
train_accuracies = []
val_accuracies = []

# Early stopping criteria
patience = 100
best_val_loss = float('inf')
early_stopping_counter = 0

for epoch in range(num_epochs):
    model.train()
    correct_train, total_train = 0, 0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    train_acc = 100 * correct_train / total_train

    # Validation phase
    model.eval()
    correct_val, total_val = 0, 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_acc = 100 * correct_val / total_val

    # Store accuracies
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")

    # Early stopping
    if val_acc > best_val_loss:
        best_val_loss = val_acc
        early_stopping_counter = 0
        torch.save(model.state_dict(), os.path.join(dataset_path, "lenet5_best_model.pth"))
    else:
        early_stopping_counter += 1
        if early_stopping_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load(os.path.join(dataset_path, "lenet5_best_model.pth")))

# Save Train and Validation Accuracy to CSV
accuracy_data = pd.DataFrame({
    "Epoch": list(range(1, len(train_accuracies) + 1)),
    "Train Accuracy (%)": train_accuracies,
    "Validation Accuracy (%)": val_accuracies
})

csv_file_path = os.path.join(dataset_path, "train_validation_accuracy.csv")
accuracy_data.to_csv(csv_file_path, index=False)
print(f"Train and validation accuracy saved to {csv_file_path}")

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
