import ml_helper
import torch
import os
from torch import Tensor
import numpy as np
from Bio.Seq import Seq
import torch.nn as nn
import torch.optim as optim



# NOTE: Questions
# Why all codons as output why not 6?

# NOTE: command for development
# Check if CUDA (GPU support) is available
# if torch.cuda.is_available():
#     # GPU is available
#     device = torch.device("cuda")
#     print("GPU is available. Using GPU for computations.")
# else:
#     # GPU is not available, fallback to CPU
#     device = torch.device("cpu")
#     print("GPU is not available. Using CPU for computations.")

# NOTE: start this script from the skripts directory
# check from which directory the script is run
print(os.getcwd())


# inherit from codon dataset
class CodonDataset_CNN(ml_helper.CodonDataset):
    def __init__(self, organism: str):
        super().__init__(organism)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        return data
    
    def __len__(self):
        return super().__len__()
    """
    # def transform_input(self, look_back: int, look_forward: int):
    #     inp_matrix = []
    #     for aa_matrix in self.aa_sequence:
    #         # for a_idx, a_pos in enumerate(aa_matrix):
    #         #     # get length of the amino acid sequence
    #         #     aa_seq_len = len(aa_matrix)
    #         #     a_rel_idx = a_idx / aa_seq_len
    #         #     # add the rel index to the a_pos
    #         #     a_pos = torch.cat((a_pos, torch.tensor([a_rel_idx])))
    #         inp_sec = []
    #         for a_idx, a_pos in enumerate(aa_matrix):
    #             # TODO: zeros or -1?
    #             # check if the index is in the look back or look forward range if outside add list with zeros
    #             prev_aa = [0] * len(a_pos) * look_back
    #             next_aa = [0] * len(a_pos) * look_forward
    #             if a_idx - look_back >= 0:
    #                 prev_aa = aa_matrix[a_idx - look_back: a_idx]
    #             if a_idx + look_forward < len(aa_matrix):
    #                 next_aa = aa_matrix[a_idx + 1: a_idx + look_forward + 1]
    #             # concatenate the previous and next amino acids with the current amino acid in the middle
    #             inp_sec.append(torch.cat((prev_aa, a_pos, next_aa)))
    #         inp_matrix.append(inp_sec)
    #     return torch.from_numpy(np.array(inp_matrix))
    """

    def transform_input(self, look_back: int, look_forward: int) -> Tensor:
        min_len = 200
        max_len = 500

        x_data = []
        y_data = []
        # create a input window like a rowlling window
        for idx, row in self.df.iterrows():
            seq = row["translation"].seq
            #if seq is over 400 or under 300 skip the sequence
            if len(seq) >= max_len or len(seq) < min_len:
                continue
            aa_sequence = ml_helper.aa_to_onehot_tensor(seq)
            
            # calculate the missing padding at the end to get to max length
            forward_padding = max_len - len(aa_sequence) + look_forward
            # TODO: zeros or -1?
            padded_aa_sequence = np.pad(aa_sequence, ((look_back, forward_padding), (0, 0)), 'constant', constant_values=0)
            # split the padded aa sequence into a parts of rows with the length of look_back + 1 + look_forward
            aa_sequence = [padded_aa_sequence[i:i + look_back + 1 + look_forward] for i in range(len(padded_aa_sequence) - look_forward - look_back)]
            aa_sequence = torch.tensor(np.array(aa_sequence))
            x_data.append(aa_sequence)

            codon_sequence = row["sequence"]
            codon_sequence = ml_helper.codon_to_tensor(codon_sequence)
            # add the padded codon sequence to the y_data
            # TODO: padding categories are len(ml_helper.codons) or -1
            forward_padding= forward_padding - look_forward
            padded_codon_sequence = np.pad(codon_sequence, (0, forward_padding), 'constant', constant_values=-1)
            y_data.append(torch.tensor(padded_codon_sequence))

        x_data = torch.stack(x_data)
        y_data = torch.stack(y_data)
        return x_data, y_data



            



# Create the dataset for a specific organism
organism = "E.Coli"  # replace with the organism you want

dataset = CodonDataset_CNN(organism)

l_b = 0
l_f = 0
input_dataset = dataset.transform_input(look_back=l_b, look_forward=l_f)
sample_size = len(input_dataset[0])
print( f"Number of samples in dataset: {sample_size}")
#print(f"First sample in the dataset: {input_dataset[0]}")


# shuffle the dataset
shuffled_indices = torch.randperm(sample_size)
x_data = input_dataset[0][shuffled_indices]
y_data = input_dataset[1][shuffled_indices]
print(f"shape of x_data: {x_data.shape}")
# print shape of the dataset
#if l_b == 0 and l_f == 0:
# flat input
if l_b == 0 and l_f == 0:
    x_data = x_data.squeeze(dim=2)
    print(f" Flattend Shape of the dataset: {x_data.shape}")

# split the dataset into training and validation sets
train_size = int(0.8 * sample_size)
val_size = sample_size - train_size
x_train = x_data[:train_size]
y_train = y_data[:train_size]
x_val = x_data[train_size:]
y_val = y_data[train_size:]

# print y shape
print(f"shape of y_data: {y_data.shape}")

hidden_size = 64
output_size = 64
    





class CNNModel(nn.Module):
    def __init__(self, input_size, output_size, num_filters):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(num_filters, output_size, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = self.softmax(x)
        return x.permute(0, 2, 1)
    """
    def __init__(self, input_size, output_size, num_filters):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_size, num_filters, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_filters, output_size, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = self.softmax(x)
        return x.permute(0, 2, 1)
    """

# Define hyperparameters
input_size = x_train.size(2)  # Size of one-hot encoded vector
output_size = y_train.max().item() + 1  # Number of unique classes in output array
num_filters = 64  # Number of filters in convolutional layer

# Create model instance
model = CNNModel(input_size, output_size, num_filters)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 10
for epoch in range(epochs):

    optimizer.zero_grad()
    outputs = model(x_train.float())  # outputs should have shape [batch_size, num_classes, sequence_length]
    print('output shape',outputs.shape)
    loss = criterion(outputs.permute(0, 2, 1), y_train)    # No need to reshape y_train

    """
    optimizer.zero_grad()
    outputs = model(x_train.float())
    outputs = outputs.unsqueeze(1)
    loss = criterion(outputs.permute(0, 2, 1), y_train) 
    """
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")








"""

class CNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_channels=21, out_channels=64, kernel_size=5, padding=2)
        # add max pooling layer
        #self.pool = torch.nn.MaxPool1d(2)
        # add flatten layer
        self.conv2 = torch.nn.Conv1d(in_channels=64, out_channels=num_classes, kernel_size=1)

        self.pool_1 = torch.nn.MaxPool1d(2)
        self.pool_2 = torch.nn.MaxPool1d(2)
        # add softmax layer
        self.flatten = torch.nn.Flatten()
        # add dense layer
        self.fc = torch.nn.Linear(64, num_classes)
        self.softmax = torch.nn.Softmax(dim=1)


    def forward(self, x):
        x = x.view(x.shape[0], x.shape[2], x.shape[1])  # Reshape the input tensor
        x = x.float()  # Convert the input tensor to float
        x = self.conv1(x)  # Apply the first convolutional layer
        x = torch.relu(x)  # Apply the ReLU activation function
        x = self.conv2(x)  # Apply the second convolutional layer
        x = torch.relu(x)  # Apply the ReLU activation function
        x = self.pool_1(x)
        x = self.pool_2(x)
        x = self.flatten(x)  # Flatten the tensor
        x = self.fc(x)  # Apply the dense layer
        x = self.softmax(x)  # Apply the softmax layer
        return x
    
num_classes = len(ml_helper.codons)
model = CNN(num_classes)
criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
batch_size = 64

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

accuracies = []
losses = []
for epoch in range(num_epochs):
    model.train()
    for i, (x_batch, y_batch) in enumerate(train_loader):
        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch.long())  # Ensure y_batch is of type long
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0
        for x_batch, y_batch in val_loader:
            y_pred = model(x_batch)
            val_loss += criterion(y_pred, y_batch).item()

            # Calculate the accuracy
            _, predicted_class_indices = torch.max(y_pred, dim=1)
            correct_predictions += (predicted_class_indices == y_batch).sum().item()
            total_predictions += y_batch.size(0)

        accuracy = correct_predictions / total_predictions
        accuracies.append(accuracy)
        losses.append(val_loss / len(val_loader))
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {val_loss / len(val_loader)}, Accuracy: {accuracy}")


# plot the loss and acc in sublots over the epochs
import matplotlib.pyplot as plt

fig, ax = plt.subplots(2)
ax[0].plot(range(num_epochs), losses, label="Loss")
ax[0].set_title("Loss over epochs")
ax[0].set_xlabel("Epoch")
ax[0].set_ylabel("Loss")
ax[0].legend()

ax[1].plot(range(num_epochs), accuracies, label="Accuracy")
ax[1].set_title("Accuracy over epochs")
ax[1].set_xlabel("Epoch")
ax[1].set_ylabel("Accuracy")
ax[1].legend()

plt.tight_layout()
plt.show()

"""