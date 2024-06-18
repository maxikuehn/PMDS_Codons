import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import sys
import numpy as np
import random
from sklearn.metrics import accuracy_score

import ml_helper as mlh
import Classifier
import Baseline_classifiers as bc


class TemporalConvNet(nn.Module):
    def __init__(self, num_features: int, num_classes: int, num_filters: int, filter_size: int, 
                 dropout_factor: float, num_blocks: int, parallel_layer: bool) -> None:
        """
        This function initializes the TemporalConvNet model
        ------
        num_features: number of features in the input
        num_classes: number of classes to predict
        num_filters: number of filters in each convolutional layer
        filter_size: size of the convolutional filter
        dropout_factor: dropout factor
        num_blocks: number of blocks of dilated convolutions
        parallel_layer: whether to add the parallel layer
        """
        super(TemporalConvNet, self).__init__()
        self.num_blocks = num_blocks
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.dropout_factor = dropout_factor
        self.parallel_layer = parallel_layer
        if self.parallel_layer:
            self.optional_layer = nn.Conv1d(num_features, num_filters, kernel_size=1)

        self.layers = nn.ModuleList()
        # Input layer
        self.layers.append(nn.Conv1d(num_features, num_filters, kernel_size=1))
        self.layers.append(nn.BatchNorm1d(num_filters))
        
        # Dilated convolutional blocks
        for i in range(num_blocks):
            dilation = 2 ** i
            # this function defines the padding and therefore teh sequence length
            pad = (filter_size - 1) * dilation // 2 
            # TODO: fix for even filter size
            self.layers.append(nn.Conv1d(num_filters, num_filters, kernel_size=filter_size, dilation=dilation, padding=pad))
            self.layers.append(nn.BatchNorm1d(num_filters))
            # if last than not dropout
            if i != num_blocks - 1:
                self.layers.append(nn.Dropout(dropout_factor))

        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_factor))

        self.layers.append(nn.Softmax(dim=1))

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        This function defines the forward pass of the model
        ------
        x: input tensor of shape (batch_size, num_features, sequence_length)
        ------
        returns: output tensor of shape (batch_size, num_classes, sequence_length)
        """
        x_init = x.permute(0, 2, 1)
        x = x_init.clone()
        for layer in self.layers:
            x = layer(x)
        # skipped all other connections and add it at the end
        if self.parallel_layer:
            x_optional = self.optional_layer(x_init)
            x = x + x_optional

        return x
    
def train(model, training_data: DataLoader, valid_data: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module, device: str) -> None:
    """
    Trains the model for the specified number of epochs
    ------
    model: TCN model
    training_data: DataLoader with the training data
    valid_data: DataLoader with the valid data
    epochs: number of epochs to train the model
    optimizer: optimizer to use
    loss_fn: loss function to use
    ------
    Returns the training losses and accuracies
    """
    trainings_losses = []
    trainings_accuracies = []
    valid_avg_accs = []

    model.to(device)
    model.train()

    for epoch in range(epochs):
        accuracies = []
        losses = []

        # Training loop
        for inputs, labels in training_data:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss = loss_fn(outputs.squeeze(), labels.squeeze().long())
            loss.backward()
            optimizer.step()

            # get accuracy
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.view(-1)
            labels = labels.view(-1)
            acc = accuracy_score(labels.cpu(), predicted.cpu())
            accuracies.append(acc)

            losses.append(loss.item())
        
        # Validation loop
        model.eval()  # Set the model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            valid_accs = []
            for valid_in, valid_labels in valid_data:
                valid_in = valid_in.to(device)
                valid_labels = valid_labels.to(device)

                valid_out = model(valid_in)

                # get accuracy
                _, valid_pred = torch.max(valid_out, 1)
                valid_pred = valid_pred.view(-1)
                valid_labels = valid_labels.view(-1)
                valid_acc = accuracy_score(valid_labels.cpu(), valid_pred.cpu())
                valid_accs.append(valid_acc)
        
        model.train()  # Set the model back to training mode

        epoch_loss = np.mean(losses)
        epoch_acc = np.mean(accuracies)
        epoch_valid_acc = np.mean(valid_accs)

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {round(epoch_loss, 3)}, accuracy: {round(epoch_acc, 3)}, valid acc: {round(epoch_valid_acc, 3)}')

        trainings_losses.append(epoch_loss)
        trainings_accuracies.append(epoch_acc)
        valid_avg_accs.append(epoch_valid_acc)


    return trainings_losses, trainings_accuracies, valid_avg_accs

def evaluate_model(model, device, test_loader: DataLoader) -> list:
    """
    This function evaluates the model on the test data
    ------
    model: model to evaluate
    test_loader: DataLoader with the test data
    ------
    returns: list with the predictions and labels
    """
    model.eval()  # Set the model to evaluation mode
    accuracies = []

    with torch.no_grad():
        for input_data, labels in test_loader:

            input_data = input_data.to(device)
            labels = labels.to(device)

            labels = labels.view(-1)

            # Forward pass
            outputs = model(input_data)
            
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.view(-1)

            # Compute custom metrics
            accuracy = accuracy_score(predicted.cpu(), labels.cpu())
            accuracies.append(accuracy)

    # Compute average accuracy
    
    return predicted, labels, accuracies

def predict(model, device, input_data: torch.Tensor):
    input_data = input_data.to(device)
    # Forward pass
    outputs = model(input_data)
    
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.view(-1)
    return predicted




# ---------------- Wrapping in Classifier Class ----------------

def prepare_aa_sequence(aa_sequence, padding_pos='right', device='cpu'):
    max_length = 500
    #non_cut_aa_sequence = np.array(aa_sequence).copy()
    #non_cut_aa_sequence = mlh.aa_to_int_tensor(aa_sequence, device)
    non_cut_aa_sequence = np.array(aa_sequence, dtype=object)
    #non_cut_aa_sequence = aa_sequence.numpy()
    # seq to tensor
    #non_cut_aa_sequence = np.array(aa_sequence) # .clone() #np.array(aa_sequence).copy()
    aa_sequences, bit_map = mlh.cut_sequence(non_cut_aa_sequence, max_length)
    for i, aa_sequence in enumerate(aa_sequences):
        aa_sequences[i] = mlh.pad_tensor(aa_sequence[0], max_length, mlh.aminoacids_to_integer['_'], padding_pos)

    return aa_sequences, bit_map, non_cut_aa_sequence


def predict_codons(model, aa_sequence_list, device):
    """ TODO currntly only with dataloader
    if not isinstance(aa_sequence_list, list):
        for idx, seq in enumerate(aa_sequence_list):
            aa_sequence_list[idx] = seq[0]
    """
    # Prepare data (pad, convert to tensor)
    prepared_amino_seq = []
    cut_bit_map = ""
    for seq in aa_sequence_list:
        #aa_sequences, bit_map, _ = prepare_aa_sequence(seq)
        #print(len(aa_sequences[0][0]), len(seq[0][0]))
        #prepared_amino_seq += aa_sequences
        #cut_bit_map += bit_map
        prepared_amino_seq.append(seq[0])
        
    
    for idx, seq in enumerate(prepared_amino_seq):
        prepared_amino_seq[idx] = mlh.aa_int_to_onehot_tensor(seq)
    #one_hot_data = mlh.aa_int_to_onehot_tensor(prepared_amino_seq, device)

    #data_loader = DataLoader(prepared_amino_seq, batch_size=len(prepared_amino_seq),  shuffle=False)

    model.eval()
    codon_predictions = []

    with torch.no_grad():
        for aa_seq in prepared_amino_seq:
            input_data = aa_seq.to(device)
            # Forward pass
            outputs = model(input_data)
            
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.view(-1)

            #pred_codons = [str(mlh.integer_to_codons[c.item()]) for c in predicted]
            pred_codons = [int(c.item()) for c in predicted]
            codon_predictions.append(pred_codons)

            """
            predicted_codons = []
            for seq_i in range(output.shape[1]):
                aa_num = batch[seq_i].item()
                if aa_num == mlh.aminoacids_to_integer['_']:
                    continue
                codon_idx = torch.argmax(output[seq_i]).item()
                codon = mlh.integer_to_codons[codon_idx]
                predicted_codons.append(codon)
                codon_predictions.append(predicted_codons)
            """
    # codon_predictions = mlh.rebuild_sequences(codon_predictions, cut_bit_map)
    # assert len(aa_sequence_list) == len(codon_predictions)
    return codon_predictions


class Tcn_Classifier(Classifier.Classifier):
    def __init__(self, trained_model, seed=42):
        self.model = trained_model
        super().__init__(seed)


    def predict_codons(self, aa_sequences: list, device=torch.device("cuda")):
        predictions_list = predict_codons(self.model, aa_sequences, device=device)
        #predictions_matrix = self.pad_and_convert_seq(predictions_list)
        #return predictions_matrix
        return predictions_list
    

if __name__ == "__main__":
    import ml_helper
    import ml_evaluation

    pad_int = ml_helper.codons.index('___')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for (org, segment_size) in [("E.Coli", 10), ("Drosophila.Melanogaster", 10), ("Homo.Sapiens", 10)]:
        test_dataset = ml_helper.CodonDataset(organism=org, split="test", padding_pos=None, data_path='./data/', one_hot_aa=False)
        test_loader = DataLoader(test_dataset, batch_size=1)
        print(f"Datensatz geladen für {org}")

        # test all models and save the best one for further evaluation
        tcnn_Model = ml_helper.load_model("tcn", org, device=device, path_model_dir="./ml_models")

        #classifier = rnn.RNN_Classifier(rnnModel)
        classifier = Tcn_Classifier(tcnn_Model)
        preds = classifier.predict_codons(test_loader)

        predictions = []
        for pred in preds:
            pred_new = [int(p) if p != '___' and p != '' else int(pad_int) for p in pred]
            predictions.append(pred_new)
        

        labels = []
        for seq, lab in test_dataset:
            # lab = seq_lab[1]
            #print(lab)
            #lab = [str(mlh.integer_to_codons[c.item()]) for c in lab]
            lab = [int(c.item()) for c in lab]
            labels.append(lab)

        for i, (lab, pred) in enumerate(zip(labels, predictions)):
            if len(lab) != len(pred):
                print(f"Mismatch at index {i}: labels length {len(lab)}, predictions length {len(pred)}")
        
        # labels = test_dataset[:][1]
        seg_acc, seg_el = classifier.calc_accuracy_per_segment(labels, predictions, segment_size=segment_size)

        #seg_acc, seg_el = classifier.calc_accuracy_per_segment(labels, predictions, segment_size=segment_size, cut_off=0.25)
        ml_evaluation.plot_accuracies_per_segment(seg_acc, seg_el, f"Accuracy pro Segment mit einer Segmentgröße von {segment_size}\nRNN Modell und Daten für {org}")