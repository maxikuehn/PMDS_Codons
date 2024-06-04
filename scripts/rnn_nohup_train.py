from datetime import datetime
import rnn
import ml_helper as mlh
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

print("\n\n")
print("=" * 80)
print("NOHUP SESSION", datetime.now())

# Set up device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
device = torch.device("cpu")  # RNN training way faster on CPU
print(f"Using {device} device")


# Prepare data
organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]
organism = organisms[2]
batch_size = 1
min_length = None
max_length = None
padding_pos = "right" if batch_size > 1 else None

train_dataset = mlh.CodonDataset(organism=organism, split="train", min_length=min_length, max_length=max_length, padding_pos=padding_pos)
print("Loaded data for", organism)
print("Länge train_dataset:", len(train_dataset))
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=8)


# Model
input_dim = len(mlh.amino_acids)
output_dim = len(mlh.codons)
n_hidden = 128

rnnModel = rnn.RNN(input_size=input_dim, hidden_size=n_hidden, output_size=output_dim, batch_size=batch_size)
print(rnnModel)

# Train variables
epochs = 1
learning_rate = 0.001
loss = nn.CrossEntropyLoss()
# optimizer = optim.Adagrad(rnnModel.parameters(), lr=learning_rate)
# optimizer = optim.Adam(rnnModel.parameters(), lr=learning_rate)
# optimizer = optim.SGD(rnnModel.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.RMSprop(rnnModel.parameters(), lr=learning_rate)


for i in range(2):
    # Start Training
    rnn.train(rnnModel, data=train_loader, epochs=epochs, optimizer=optimizer, loss_fn=loss, device=device)

    mlh.save_model(rnnModel, "rnn", organism, appendix=f"hidden{n_hidden}_epochs{epochs+i}_lr{learning_rate}_optim{optimizer.__class__.__name__}")
