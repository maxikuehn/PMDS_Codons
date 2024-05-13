import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import ml_helper
import torch
import os
from torch import Tensor
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import seaborn as sns
import sys
import torch.nn.functional as F

sys.path.append('../scripts')


organism = "E.Coli"
min_length = 100
max_length = 500

train_dataset = ml_helper.CodonDataset(organism, "train", min_length, max_length, one_hot_aa=True)
print(f"Länge train_dataset: {len(train_dataset)}")
test_dataset = ml_helper.CodonDataset(organism, "test", min_length, max_length, one_hot_aa=True)
print(f"Länge test_dataset: {len(test_dataset)}")


# Define the ConvS2S model
class ConvS2S(nn.Module):
    def __init__(self, num_features, num_filters, num_output):
        super(ConvS2S, self).__init__()
        self.conv1 = nn.Conv1d(num_features, num_filters, kernel_size=1)
        self.attention = nn.Linear(num_filters, 1)
        self.feedforward = nn.Linear(num_filters, num_output)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = F.glu(self.conv1(x))
        attention_weights = F.softmax(self.attention(x.transpose(1, 2)), dim=-1)
        attention_weights = attention_weights.unsqueeze(1)  # Add an extra dimension to the attention weights
        x = torch.sum(attention_weights * x, dim=2)  # Multiply the attention weights and the output of the convolutional layer
        x = x.squeeze(1)
        x = self.feedforward(x)
        return x

# Initialize the model
num_features = len(ml_helper.amino_acids)
num_classes = 64
num_filters = 64
model = ConvS2S(num_features, num_filters, num_classes)

# Create DataLoader
BATCH_SIZE = 32
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)


# Get a single batch of data
inputs, targets = next(iter(train_loader))

# Print the shapes
print('Inputs shape:', inputs.shape)
print('Targets shape:', targets.shape)

criterion = nn.CrossEntropyLoss(ignore_index=64)
optimizer = optim.Adam(model.parameters(), lr=0.005) # lr=0.001

# Training loop
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {running_loss}')





from sklearn.metrics import accuracy_score

def compute_accuracy(predictions, labels):
    acc = accuracy_score(labels, predictions)
    return acc


def evaluate_model(model, test_loader):
    model.eval()  # Set the model to evaluation mode

    accuracies = []

    with torch.no_grad():
        for input_data, labels in test_loader:
            labels = labels.view(-1)

            # Forward pass
            outputs = model(input_data.permute(0, 2, 1))
            
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.view(-1)

            # Compute custom metrics
            accuracy = compute_accuracy(predicted.cpu(), labels.cpu())
            accuracies.append(accuracy)

    # Compute average accuracy
    
    return predicted, labels, accuracies



predicted, labels, accuracies = evaluate_model(model, test_loader)

avg_accuracy = np.mean(accuracies)

# from tensor to normal numbers for each position
predicted = predicted.tolist()
# print unique elements form list
unique_elements = set(predicted)
print(unique_elements)

labels = labels.tolist()
uni_lab = set(labels)
print(uni_lab)


from sklearn.metrics import confusion_matrix
# remove pairs where label is 64
predicted, labels = zip(*[(pred, label) for pred, label in zip(predicted, labels) if label != 64])

# Calculate confusion matrix
conf_matrix = confusion_matrix(labels, predicted, normalize='true')
mask = np.array(conf_matrix)
# Plot the confusion matrix
plt.figure(figsize=(10,7))
sns.heatmap(conf_matrix, annot=mask, cmap='coolwarm', fmt='.2f')
plt.title('Confusion Matrix of preds and labels')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
