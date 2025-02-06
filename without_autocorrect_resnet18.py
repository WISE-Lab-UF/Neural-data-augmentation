import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
from jiwer import wer

# Set device for PyTorch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define labels for characters
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z',
          'greaterThan', 'comma', 'apostrophe', 'tilde', 'questionMark']

# Define dataset path
dataset_path = os.path.expanduser('~') + '/custom/dataset/augmentation/'
test_path = os.path.join(dataset_path, "test")

# Define ResNet18-based Model
class ResNet18Custom(nn.Module):
    def __init__(self, num_classes=31):
        super(ResNet18Custom, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        self.base_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.base_model.fc = nn.Linear(self.base_model.fc.in_features, num_classes)

    def forward(self, x):
        return self.base_model(x)

# Load the model
num_classes = len(labels)
model = ResNet18Custom(num_classes=num_classes).to(device)

# Load pre-trained weights
state_dict = torch.load(os.path.join(dataset_path, "resnet18_model_only_on_training.pth"),
                        map_location=device)
model.load_state_dict(state_dict)
model.eval()

# Function to process sentences into corresponding character arrays
def process_sentence(sentence):
    word = []
    for s in sentence.lower():
        if s == ' ':
            word.append('greaterThan')
        elif s == '?':
            word.append('questionMark')
        elif s == '.':
            word.append('tilde')
        elif s == ',':
            word.append('comma')
        elif s == '\'':
            word.append('apostrophe')
        elif s == '!':
            word.append('tilde')
        else:
            word.append(s)
    return word

# Load sentences from a file
file_path = 'english.txt'
with open(file_path, 'r') as file:
    sentences = file.readlines()

# Evaluation Metrics
wer_avg = 0
cer_avg = 0
total_sentences = len(sentences)

# Store total characters, words, and time for overall efficiency metrics
total_chars = 0
total_words = 0
total_time = 0.0

# Process and evaluate each sentence
for count, line in enumerate(sentences):
    sentence = line.strip()
    word = process_sentence(sentence)
    print(f"\nProcessing sentence {count+1}/{total_sentences}: {sentence}")
    
    # Construct input tensor from character images
    for i, letter in enumerate(word):
        letter_data = np.load(os.path.join(test_path, f"{letter}.npy"))
        sample_idx = random.randint(0, min(350, letter_data.shape[0] - 1))
        char_img = letter_data[sample_idx]

        if i == 0:
            a = char_img
        else:
            a = np.append(a, char_img)
        a = a.reshape(i + 1, 224, 224)

    # Convert to PyTorch tensor
    a = torch.tensor(a, dtype=torch.float32).unsqueeze(1).to(device)

    # Run inference
    start_time = time.time()
    with torch.no_grad():
        y_pred = model(a)
    end_time = time.time()

    # Compute inference time
    duration = end_time - start_time
    print(f"Inference time: {duration:.6f} seconds")

    # Calculate Characters Per Minute (CPM) and Words Per Minute (WPM)
    num_chars = len(word)  # Number of characters in sentence
    num_words = len(sentence.split())  # Number of words
    total_chars += num_chars
    total_words += num_words
    total_time += duration

    cpm = (num_chars / duration) * 60
    wpm = (num_words / duration) * 60

    print(f"Characters per minute (CPM): {cpm:.2f}")
    print(f"Words per minute (WPM): {wpm:.2f}")

    # Function to get max index prediction
    def maxNumber(arr):
        return np.argmax(arr)

    # Convert predicted labels to string
    predicted_sentence = ""
    for i in range(len(y_pred)):
        predicted_char = labels[maxNumber(y_pred[i].cpu().numpy())]
        if predicted_char == 'greaterThan':
            predicted_sentence += ' '
        elif predicted_char == 'questionMark':
            predicted_sentence += '?'
        elif predicted_char == 'apostrophe':
            predicted_sentence += '\''
        elif predicted_char == 'comma':
            predicted_sentence += ','
        elif predicted_char == 'tilde':
            predicted_sentence += '.'
        else:
            predicted_sentence += predicted_char

    # Compute WER and CER
    def cer(ref, hyp):
        d = [[0 for j in range(len(hyp) + 1)] for i in range(len(ref) + 1)]
        for i in range(len(ref) + 1):
            d[i][0] = i
        for j in range(len(hyp) + 1):
            d[0][j] = j
        for i in range(1, len(ref) + 1):
            for j in range(1, len(hyp) + 1):
                if ref[i - 1] == hyp[j - 1]:
                    d[i][j] = d[i - 1][j - 1]
                else:
                    d[i][j] = min(d[i - 1][j - 1] + 1, d[i][j - 1] + 1, d[i - 1][j] + 1)
        return d[len(ref)][len(hyp)] / float(len(ref))

    ref_sentence = "".join(word).replace('greaterThan', ' ').replace('questionMark', '?').replace('apostrophe', '\'').replace('comma', ',').replace('tilde', '.')

    print(f"Actual: {ref_sentence}")
    print(f"Predicted: {predicted_sentence}")

    wer_value = wer(ref_sentence, predicted_sentence)
    cer_value = cer(ref_sentence, predicted_sentence)
    
    wer_avg += wer_value
    cer_avg += cer_value

    print(f"WER: {wer_value:.4f}, CER: {cer_value:.4f}")

# Print average results
average_cpm = (total_chars / total_time) * 60 if total_time > 0 else 0
average_wpm = (total_words / total_time) * 60 if total_time > 0 else 0

print(f"\nAverage WER: {wer_avg / total_sentences:.4f}")
print(f"Average CER: {cer_avg / total_sentences:.4f}")
print(f"Average Characters Per Minute (CPM): {average_cpm:.2f}")
print(f"Average Words Per Minute (WPM): {average_wpm:.2f}")
