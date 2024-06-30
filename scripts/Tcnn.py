import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score
import time
from itertools import product
import json
import os
import copy
import random
import sys

import ml_helper as mlh
import Classifier


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
    
def train(model, training_data: DataLoader, valid_data: DataLoader, epochs: int, optimizer: optim.Optimizer, loss_fn: nn.Module, device: str, 
        pad_str: str = '___', stop_area:int = 7, verbose=True) -> None:
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
    best_model = None
    best_acc = -1
    best_epoch = -1

    epoch = 0
    still_training = True
    trainings_losses = []
    trainings_accuracies = []
    valid_avg_accs = []
    pad_int = mlh.codons.index(pad_str)

    model.to(device)
    model.train()
    
    #for epoch in range(epochs):
    while still_training:
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

            # filter out padding where labels are pad_int
            mask = labels != pad_int
            labels = labels[mask]
            predicted = predicted[mask]

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

                # filter out padding where labels are pad_int
                mask = valid_labels != pad_int
                valid_labels = valid_labels[mask]
                valid_pred = valid_pred[mask]


                valid_acc = accuracy_score(valid_labels.cpu(), valid_pred.cpu())
                valid_accs.append(valid_acc)
        
        model.train()  # Set the model back to training mode

        epoch_loss = np.mean(losses)
        epoch_acc = np.mean(accuracies)
        epoch_valid_acc = np.mean(valid_accs)

        trainings_losses.append(epoch_loss)
        trainings_accuracies.append(epoch_acc)
        valid_avg_accs.append(epoch_valid_acc)

        if epoch_valid_acc > best_acc:
            best_model = copy.deepcopy(model)
            best_acc = epoch_valid_acc
            best_epoch = epoch + 1

        if verbose:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {round(epoch_loss, 3)}, accuracy: {round(epoch_acc, 3)}, valid acc: {round(epoch_valid_acc, 3)}')

        epoch += 1
        if epoch >= epochs:
            still_training = False
            if verbose:
                print("Training completed.")
        # if epochs is not None:
        #     if verbose:
        #         print(f'Epoch [{epoch+1}/{epochs}], Loss: {round(epoch_loss, 3)}, accuracy: {round(epoch_acc, 3)}, valid acc: {round(epoch_valid_acc, 3)}')
        #     if epoch == epochs:
        #         still_training = False
        # else:
        #     if verbose:
        #         print(f'Epoch [{epoch+1}], Loss: {round(epoch_loss, 3)}, accuracy: {round(epoch_acc, 3)}, valid acc: {round(epoch_valid_acc, 3)}')
            # if no improvment in last 3 epochs compared to last 7 epochs, stop training
            # if len(valid_avg_accs) > (stop_area - 1):
            #     avg_last_prog = np.mean(valid_avg_accs[-3:])
            #     avg_last_area = np.mean(valid_avg_accs[-stop_area:])
            #     if avg_last_prog < avg_last_area:
            #         still_training = False
            #         if verbose:
            #             print(f"Early stopping due to no progress in last 3 epochs compared to last 7 epochs. Last 3: {avg_last_prog}, Last 7: {avg_last_area}")

    model = best_model
    if verbose:
        print('Best Validaton Accuracy:', best_acc)
        
    return trainings_losses, trainings_accuracies, valid_avg_accs, best_epoch

def evaluate_model(model, device:str, test_loader: DataLoader,  pad_str: str= '___', codon_names=True) -> list:
    """
    This function evaluates the model on the test data
    ------
    model: model to evaluate
    test_loader: DataLoader with the test data
    device: device to use
    pad_str: padding string
    codon_names: whether to return the codon names or the integers indicating the codons
    ------
    returns: list with the predictions and labels
    """
    pad_int = mlh.codons.index(pad_str)
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

            # filter out padding where labels are pad_int
            mask = labels != pad_int
            labels = labels[mask]
            predicted = predicted[mask]

            # Compute custom metrics
            accuracy = accuracy_score(predicted.cpu(), labels.cpu())
            accuracies.append(accuracy)

    if codon_names:
        predicted = [mlh.integer_to_codons[c.item()] for c in predicted]
        labels = [mlh.integer_to_codons[c.item()] for c in labels]
    
    return predicted, labels, accuracies


def predict(model, input_data: torch.Tensor, device: str=None) -> torch.Tensor:
    """
    This function predicts the output of the model
    ------
    model: model to use
    input_data: input data
    device: device to use
    ------
    returns: predicted output
    """
    input_data = input_data.to(device)
    # Forward pass
    outputs = model(input_data)
    
    _, predicted = torch.max(outputs, 1)
    predicted = predicted.view(-1)
    return predicted


def hyperparam_search(organism:str, train_loader: DataLoader, valid_loader: DataLoader, dict_params: dict, num_features, num_classes, device, criterion):
    """
    This function performs a grid search over the hyperparameters
    ------
    organism: organism to perform the search for
    train_loader: DataLoader with the training data
    valid_loader: DataLoader with the validation data
    dict_params: dictionary with the hyperparameters to search over
    num_features: number of features in the input
    num_classes: number of classes to predict
    device: device to use
    criterion: loss function
    ------
    returns: list with the results of the search and the best model
    """

    # grid search
    results = []
    best_model = None

    hyper_params_combis = list(product(*dict_params.values()))
    
    # Data preparation
    test_dataset = mlh.CodonDataset(organism=organism, split="test", padding_pos=None, one_hot_aa=False)
    test_loader = DataLoader(test_dataset, batch_size=1)
    print(f"Datensatz geladen für {organism}")

    labels = []
    for seq, lab in test_dataset:
        lab = [int(c.item()) for c in lab]
        labels.append(lab)

    for nf, fs, df, nb, pl, lr in hyper_params_combis:
        # print training model num out of total
        print(f"Training model {len(results)+1}/{len(hyper_params_combis)}")
        start_time = time.time()

        tcnModel = TemporalConvNet(num_features, num_classes, nf, fs, df, nb, pl)
        optimizer = optim.Adam(tcnModel.parameters(), lr=lr)
        train(tcnModel, train_loader, valid_loader, None, optimizer, criterion, device=device, verbose=False)

        classifier = Tcn_Classifier(tcnModel)
        predictions = classifier.predict_codons(test_loader, codon_names=False)
        predictions = classifier.pad_and_convert_seq(predictions)

        acc = classifier.calc_accuracy(labels, predictions, pad='')
        results.append([nf, fs, df, nb, pl, lr, acc])

        if acc == max([r[-1] for r in results]):
            best_model = tcnModel

        end_time = time.time()
        print(f"Model trained in {round(end_time-start_time,2)} sec, acc: {round(acc, 3)} curr best: {round(max([r[-1] for r in results]), 3)} curr worst: {round(min([r[-1] for r in results]), 3)}")
        # estimate time left in minutes
        print(f"Estimated time left: {round((end_time-start_time)*(len(hyper_params_combis)-len(results))/60, 0)} minutes")

    return results, best_model



def load_best_params(organism:str, path_hyperpara_results:str=None):
    """
    This function returns the best hyperparameters from the hyperparameter search
    ------
    organism: organism to get the best hyperparameters for
    path_hyperpara_results: path to the hyperparameter search results
    ------
    returns: best hyperparameters
    """
    if path_hyperpara_results is None:
        path = f'../ml_models/{organism}/tcnn_grid_search_results.json'
    else:
        path = path_hyperpara_results
    
    if not os.path.exists(path):
        print("No hyperparameter search results found. Plase run hyperparameter search first.")
        return None

    with open(path, 'r') as f:
        results = json.load(f)

    best_params = results[np.argmax([r[-1] for r in results])]
    # cut last element (accuracy)
    best_params = best_params[:-1]
    return best_params


# ---------------- Wrapping in Classifier Class ----------------

def predict_codons(model, aa_sequence_list, device=torch.device("cuda"), as_codon_names=False):
    """ TODO currntly only with dataloader
    if not isinstance(aa_sequence_list, list):
        for idx, seq in enumerate(aa_sequence_list):
            aa_sequence_list[idx] = seq[0]
    """
    # Prepare data (pad, convert to tensor)
    prepared_amino_seq = []
    for seq in aa_sequence_list:
        prepared_amino_seq.append(seq[0])
        
    for idx, seq in enumerate(prepared_amino_seq):
        prepared_amino_seq[idx] = mlh.aa_int_to_onehot_tensor(seq)

    model.eval()
    codon_predictions = []

    with torch.no_grad():
        for aa_seq in prepared_amino_seq:
            input_data = aa_seq.to(device)
            # Forward pass
            outputs = model(input_data)
            
            _, predicted = torch.max(outputs, 1)
            predicted = predicted.view(-1)

            # strings of codons or integers
            if as_codon_names:
                pred_codons = [mlh.integer_to_codons[c.item()] for c in predicted]
            else:
                pred_codons = [c.item() for c in predicted]
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


    def predict_codons(self, aa_sequences: list, device=torch.device("cuda"), codon_names=True):
        predictions_list = predict_codons(self.model, aa_sequences, device=device, as_codon_names=codon_names)
        #predictions_matrix = self.pad_and_convert_seq(predictions_list)
        #return predictions_matrix
        return predictions_list
    

if __name__ == "__main__":
    # cd scripts

    # nohup python3 scripts/Tcnn.py &
    # tail -f nohup.out
    # jobs -l

    # TODO: change number of blocks in grid serarch results human to 5 from 7

    data_path = './data'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    def set_seed(SEED=SEED):
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
    set_seed()

    # Data preparation
    organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]
    organism = organisms[2]
    pad_int = mlh.codons.index('___')
    #"Homo.Sapiens"  "Drosophila.Melanogaster"  "E.Coli"
    min_length = None
    max_length = 500
    one_hot = True
    cut_data = False

    SPEEDS_ADDED = False
    BATCH_SIZE = 32

    print('Start loading data for organism: ', organism)
    train_dataset = mlh.CodonDataset(organism, "train", min_length, max_length, cut_data=True, one_hot_aa=one_hot, data_path=data_path, device=device)
    print(f"Länge train_dataset: {len(train_dataset)}")
    valid_dataset = mlh.CodonDataset(organism, "valid", min_length, max_length, cut_data=True, one_hot_aa=one_hot, data_path=data_path, device=device)
    print(f"Länge valid_dataset: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

    path_params = f'./ml_models/{organism}/tcnn_grid_search_results.json'
    best_params = load_best_params(organism, path_params)
    if best_params is not None:
        print(f"Best parameters: num_filters: {best_params[0]}, filter_size: {best_params[1]}, dropout_factor: {best_params[2]}, num_blocks: {best_params[3]}, parallel_layer: {best_params[4]}, learing_rate: {best_params[5]}")

    # Hyperparameters
    num_features = len(mlh.amino_acids)
    num_classes = len(mlh.codons)  # number of codons (output classes)
    NUM_EPOCHS = 200

    num_filters = 128 #64
    filter_size = 5#3  # NOTE: filter size must be unequal like: 3,5,9,...
    dropout_factor = 0.08 #0.08 #0.005
    num_blocks = 5#2
    parallel_layer = True
    learing_rate = 0.001 # 0.001

    if best_params is not None:
        num_filters = best_params[0]
        filter_size = best_params[1]
        dropout_factor = best_params[2]
        num_blocks = best_params[3]
        parallel_layer = best_params[4]
        learing_rate = best_params[5]


    # Model
    tcnModel = TemporalConvNet(num_features, num_classes, num_filters, filter_size, 
                            dropout_factor, num_blocks, parallel_layer)
    print(tcnModel)

    #criterion = nn.CrossEntropyLoss(ignore_index=64)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_int)
    optimizer = optim.Adam(tcnModel.parameters(), lr=learing_rate)

    trainings_losses, trainings_accuracies, valid_accs, best_epoch_idx = train(tcnModel, train_loader, valid_loader, NUM_EPOCHS, optimizer, criterion, device=device)


    json_data = {'organism': organism, 'training_valid_accs': valid_accs, 'best_epoch_idx': best_epoch_idx}

    # Save training valid accuracies
    acc_path = data_path + f'/{organism}/training_valid_accs.json'

    with open(acc_path, 'w') as file:
        json.dump(json_data, file)

    print('Vaild Accs in training saved in:', acc_path)

    mlh.save_model(tcnModel, f'tcn_valid_acc_{round(round(valid_accs[-1],2) * 100)}', organism, dir_path='./ml_models')
    """        
    organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]
    organisms = [organisms[0]]

    pad_int = mlh.codons.index('___')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for org in organisms:
        test_dataset = mlh.CodonDataset(organism=org, split="test", padding_pos=None, data_path='./data/', one_hot_aa=False)
        test_loader = DataLoader(test_dataset, batch_size=1)
        print(f"Datensatz geladen für {org}")

        tcnn_Model = mlh.load_model("tcn", org, device=device, path_model_dir="./ml_models")

        classifier = Tcn_Classifier(tcnn_Model)
        preds = classifier.predict_codons(test_loader, codon_names=False)

        labels = []
        for seq, lab in test_dataset:
            lab = [int(c.item()) for c in lab]
            labels.append(lab)
        
        for i, (lab, pred) in enumerate(zip(labels, preds)):
            if len(lab) != len(pred):
                print(f"Mismatch at index {i}: labels length {len(lab)}, predictions length {len(pred)}")

    
        acc = classifier.calc_accuracy(labels, preds, pad='')
        print(f"Organismus {org} with a Accuracy: {round(acc, 3)}")
        """