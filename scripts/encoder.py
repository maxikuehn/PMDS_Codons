
from copy import deepcopy
import torch
from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn as nn
import math
import time
import random
import numpy as np
import torch.optim as optim
import pandas as pd

import ml_helper as mlh
import Classifier
import Baseline_classifiers as bc
import custom_transformer_encoder as cte

SPEEDS_ADDED = False
SEED = 42
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
data_path = '../data'

organism = ""
train_loader = None
valid_loader = None
df = None
usage_biases = None


def set_organism(_organism, BATCH_SIZE=32):
    load_train_valid_data(_organism, BATCH_SIZE)
    load_test_data(_organism)


def load_train_valid_data(_organism, BATCH_SIZE=32):
    global organism
    global train_loader
    global valid_loader

    organism = _organism

    min_length = None
    max_length = 500

    train_dataset = mlh.CodonDataset(organism, "train", min_length, max_length, add_speeds=SPEEDS_ADDED, cut_data=True, one_hot_aa=False, data_path=data_path, device=device)
    print(f"Länge train_dataset: {len(train_dataset)}")
    valid_dataset = mlh.CodonDataset(organism, "valid", min_length, max_length, add_speeds=SPEEDS_ADDED, cut_data=True, one_hot_aa=False, data_path=data_path, device=device)
    print(f"Länge valid_dataset: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)


def load_test_data(_organism):
    global organism
    global df
    global usage_biases

    organism = _organism

    df = pd.read_pickle(f"../data/{organism}/cleanedData_test.pkl")
    usage_biases = pd.read_pickle(f"../data/{organism}/usageBias.pkl")
    df['codons'] = df['sequence'].apply(group_codons)
    print(f"Länge test df: {len(df)}")


def load_shuffled_data():
    global df
    global usage_biases

    df = pd.read_pickle(f"../data/{organism}/cleanedData_test_shuffled.pkl")
    usage_biases = pd.read_pickle(f"../data/{organism}/usageBias.pkl")
    df['codons'] = df['sequence'].apply(group_codons)


def group_codons(sequence):
    return [''.join(sequence[i:i+3]) for i in range(0, len(sequence), 3)]


def set_seed(SEED=SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)
    

class EncoderClassifier(nn.Module):
    def __init__(self, embed_dim, num_layers, num_heads, dropout=0.2, pos_enc=False):
        super(EncoderClassifier, self).__init__()

        emb_size = embed_dim
        if SPEEDS_ADDED:
            emb_size -= 1
        self.emb = nn.Embedding(len(mlh.amino_acids), emb_size, padding_idx=len(mlh.amino_acids)-1)
        self.pos_enc = pos_enc
        self.pos_encoder = PositionalEncoding(embed_dim, dropout)

        self.encoder_layer = cte.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True
        )
        self.encoder = cte.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers,
        )

        self.linear = nn.Linear(embed_dim, len(mlh.codons))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_weights_needed=False):
        x = x.long()
        if SPEEDS_ADDED:
            x1 = self.emb(x[:, :, 0])
            x2 = x[:, :, 1].unsqueeze(-1)
            x = torch.cat((x1, x2), dim=-1)  # Concatenate along the feature dimension
        else:
            x = self.emb(x)

        if self.pos_enc:
            x = x.transpose(0, 1)
            x = self.pos_encoder(x)  # Add positional encoding
            x = x.transpose(0, 1)
        
        if attn_weights_needed:
            x, attn_weights = self.encoder(x, attn_weights_needed=True)
            x = self.dropout(x)
            out = self.linear(x)
            return out, attn_weights
        else:
            x = self.encoder(x)
            x = self.dropout(x)
            out = self.linear(x)
            return out


def count_correct_predictions(predictions, labels):
    predictions = np.argmax(predictions, axis=1)

    # Find indices where labels are not equal to the padding value
    non_padding_indices = labels != mlh.codons_to_integer['___']

    # Filter out predictions and labels where the label is not padding
    filtered_predictions = predictions[non_padding_indices]
    filtered_labels = labels[non_padding_indices]

    codon_num = filtered_labels.shape[0]
    correct_codons = (filtered_predictions == filtered_labels).sum().item()
    return codon_num, correct_codons


def evaluate_model(model, criterion, print_scores=True, loss_without_pad=False):
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0

    with torch.no_grad():
        codon_num = 0
        correct_codon_num = 0
        for batch_idx, batch in enumerate(valid_loader):
             # Forward pass
            input_data, labels = batch

            output = model(input_data)  # (batch_size, seq_len, num_classes)
            output = output.view(-1, len(mlh.codons)) # (batch_size * seq_len, num_classes)

            labels = labels.view(-1).long() # (batch_size, seq_len) -> (batch_size * seq_len)

            # Calculate loss
            loss = criterion(output, labels)

            # Compute total loss
            total_loss += loss.item()

            # Count codons and correct codon predictions
            codon_num_batch, correct_codons_batch = count_correct_predictions(output.cpu(), labels.cpu())
            codon_num += codon_num_batch
            correct_codon_num += correct_codons_batch

    # Compute average loss
    avg_loss = total_loss / len(valid_loader)

    # Compute accuracy
    accuracy = round(correct_codon_num / codon_num, 4)

    if print_scores:
        print(f'Average Batch Loss: {avg_loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}')

    return avg_loss, accuracy


def train_model(model, num_epochs, loss_ignore_pad=True, learning_rate=0.0005, validation_stop=True, validation_stop_area=7, print_batches=0, print_epochs=True, start_epoch=0, current_best_model_state=None):
    criterion = torch.nn.CrossEntropyLoss()
    if loss_ignore_pad:
        criterion = torch.nn.CrossEntropyLoss(ignore_index=mlh.codons_to_integer['___'])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_model_state = {
        "state": None,
        "accuracy": 0,
        "epoch": None
    }

    if current_best_model_state:
        best_model_state = current_best_model_state
    
    start_time = time.time()
    last_loss = None
    saved_accuracies = []
    all_accuracies = []
    epoch_num = start_epoch
    for epoch in range(start_epoch, num_epochs):
        epoch_num += 1
        set_seed(epoch)
        model.train()

        epoch_start_time = time.time()
        batch_start_time = time.time()
        epoch_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            input_data, labels = batch

            output = model(input_data)  # (batch_size, seq_len, num_classes)
            output = output.view(-1, len(mlh.codons)) # (batch_size * seq_len, num_classes)

            labels = labels.view(-1).long() # (batch_size, seq_len) -> (batch_size * seq_len)

            # Calculate loss
            loss = criterion(output, labels)
            epoch_loss += loss.item()

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            if print_batches != 0 and batch_idx % print_batches == (print_batches-1):
                batch_time =  round(time.time() - batch_start_time,2)
                print(f'Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Time since last batch print: {batch_time} s')
                batch_start_time = time.time()
        
        epoch_loss = round(epoch_loss / len(train_loader),4)
        last_loss = epoch_loss
        
        avg_eval_loss, accuracy = evaluate_model(model, criterion, print_scores=False)
        all_accuracies.append(accuracy)

        epoch_time = round(time.time() - epoch_start_time,2)
        if print_epochs:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss}, Eval Accuracy: {accuracy}, Took {epoch_time} s')
         
        if validation_stop:
            saved_accuracies.append(accuracy)  
            if len(saved_accuracies) == validation_stop_area+1:
                # compare accuracy to average of saved_accuracies
                # if accuracy is lower: stop early
                if np.average(np.array(saved_accuracies[validation_stop_area-1:validation_stop_area+1])) < np.average(np.array(saved_accuracies[0:validation_stop_area-2])):
                    print(f'Stopped early after epoch {epoch+1} as validation accuracy was lower than average of the last {validation_stop_area} accuracies.')
                    break
                saved_accuracies.pop(0)
        else:
            if accuracy > best_model_state["accuracy"]:
                best_model_state = {
                    "state": deepcopy(model.state_dict()),
                    "accuracy": accuracy,
                    "epoch": epoch+1
                }

    model.load_state_dict(best_model_state["state"])  
    avg_eval_loss, accuracy = evaluate_model(model, criterion, print_scores=False)

    total_time = round(time.time() - start_time,2)
    print(f'Average eval Loss: {round(avg_eval_loss,4)}, Best Eval Accuracy: {accuracy}, Took {total_time} s')
    return avg_eval_loss, accuracy, all_accuracies, epoch_num, best_model_state


def train_parameter_model(embed_dim, num_encoder_layers, num_heads, dropout, pos_enc, num_epochs, print_epochs, not_relevant=False, validation_stop=True, start_epoch=0, current_best_model_state=None, existing_model=None):
    set_seed()
    
    if existing_model:
        model = existing_model
    else:
        model = EncoderClassifier(
            embed_dim=embed_dim,
            num_layers=num_encoder_layers,
            num_heads=num_heads,
            dropout=dropout,
            pos_enc=pos_enc
        ).to(device)

    print(f"----- Start Training: {embed_dim} emb, {num_encoder_layers} layers, {num_heads} heads, {dropout} dropout, positional encoding: {pos_enc}, {num_epochs} epochs -----")
    last_loss, accuracy, all_accuracies, epoch_num, best_model_state = train_model(model, num_epochs, print_epochs=print_epochs, validation_stop=validation_stop, start_epoch=start_epoch, current_best_model_state=current_best_model_state)

    saved = False
    if last_loss >= 2:
        print(f"Did not save following model as loss was too high:")
        print(f'encoder_{embed_dim}em_{num_encoder_layers}l_{num_heads}h{"_posenc" if pos_enc else ""}_{str(dropout).replace(".","")}dr_{epoch_num}ep')
    else:
        saved = True
        mlh.save_model(model, f'encoder_{embed_dim}em_{num_encoder_layers}l_{num_heads}h{"_posenc" if pos_enc else ""}_{str(dropout).replace(".","")}dr_{epoch_num}ep', organism, not_relevant=not_relevant)
    return saved, accuracy, all_accuracies, best_model_state


def hyper_parameter_training(embed_dims, num_encoder_layers, num_heads, dropouts, pos_enc, epochs=50, print_epochs=True, validation_stop=True, start_epoch=0, current_best_model_state=None, existing_model=None):
    not_saved = []
    accuracies = {}
    all_accuracies_dict = {}
    for EMBED_DIM in embed_dims:
        for NUM_ENCODER_LAYERS in num_encoder_layers:
            for NUM_HEADS in num_heads:
                for DROPOUT in dropouts:
                    for POS_ENC in pos_enc:
                        model_name = f'encoder_{EMBED_DIM}em_{NUM_ENCODER_LAYERS}l_{NUM_HEADS}h{"_posenc" if POS_ENC else ""}_{str(DROPOUT).replace(".","")}dr_{epochs}ep'
                        saved, accuracy, all_accuracies, best_model_state = train_parameter_model(EMBED_DIM, NUM_ENCODER_LAYERS, NUM_HEADS, DROPOUT, POS_ENC, epochs, print_epochs, not_relevant=True, validation_stop=validation_stop, start_epoch=start_epoch, current_best_model_state=current_best_model_state, existing_model=existing_model)
                        accuracies[model_name] = accuracy
                        all_accuracies_dict[model_name] = all_accuracies
                        if not saved:
                            not_saved.append(model_name)
    print("------------")
    print("Not saved as loss too high:")
    print(not_saved)
    return accuracies, all_accuracies_dict, best_model_state


# ---------------- Wrapping in Classifier Class ----------------

def prepare_aa_sequence(aa_sequence, padding_pos='right'):
    max_length = 500
    non_cut_aa_sequence = mlh.aa_to_int_tensor(aa_sequence, device)
    aa_sequences, bit_map = mlh.cut_sequence(non_cut_aa_sequence, max_length)
    for i, aa_sequence in enumerate(aa_sequences):
        aa_sequences[i] = mlh.pad_tensor(aa_sequence, max_length, mlh.aminoacids_to_integer['_'], padding_pos)
        if SPEEDS_ADDED:
            aa_sequences[i] = mlh.add_speed_dimension(aa_sequences[i], device)
    return aa_sequences, bit_map, non_cut_aa_sequence


def predict_codons(model, aa_sequence_list):
    # Prepare data (pad, convert to tensor)
    prepared_amino_seq = []
    cut_bit_map = ""
    for seq in aa_sequence_list:
        aa_sequences, bit_map, _ = prepare_aa_sequence(seq)
        prepared_amino_seq += aa_sequences
        cut_bit_map += bit_map

    # create data_loader for batched throughput
    batch_size = 32
    data_loader = DataLoader(prepared_amino_seq, batch_size=batch_size)

    model.eval()
    codon_predictions = []
    with torch.no_grad():
        for batch in data_loader:
            output = model(batch)  # (batch_size, seq_len, num_classes)

            for batch_i in range(output.shape[0]):
                predicted_codons = []
                for seq_i in range(output.shape[1]):
                    if SPEEDS_ADDED:
                        aa_num = batch[batch_i][seq_i][0].item()
                    else:
                        aa_num = batch[batch_i][seq_i].item()
                    if aa_num == mlh.aminoacids_to_integer['_']:
                        continue
                    codon_idx = torch.argmax(output[batch_i][seq_i]).item()
                    codon = mlh.integer_to_codons[codon_idx]
                    predicted_codons.append(codon)
                codon_predictions.append(predicted_codons)
    codon_predictions = mlh.rebuild_sequences(codon_predictions, cut_bit_map)
    assert len(aa_sequence_list) == len(codon_predictions)
    return codon_predictions


class Encoder_Classifier(Classifier.Classifier):
    def __init__(self, trained_model, seed=42):
        self.model = trained_model
        super().__init__(seed)


    def predict_codons(self, aa_sequences, replace=False):
        predictions_list = predict_codons(self.model, aa_sequences)
        if replace:
            predictions_list = bc.check_and_replace_codons(aa_sequences, predictions_list, usage_biases)
        predictions_matrix = self.pad_and_convert_seq(predictions_list)
        return predictions_matrix
    

def eval_best_model():
    try:
        model = mlh.load_model( f'encoder', organism, device=device)
    except Exception as e:
        print(e)
        print("Not found:")
        print(f'encoder')
        return

    encoder_classifier = Encoder_Classifier(model)
    amino_seq = df['translation']
    true_codons = df['codons']
    pred_codons_replaced = encoder_classifier.predict_codons(amino_seq, replace=True)
    return true_codons, pred_codons_replaced


def eval_parameter_model(embed_dim, num_encoder_layers, num_heads, dropout, pos_enc, not_relevant=False):
    start_time = time.time()

    try:
        model = mlh.load_model( f'encoder_{embed_dim}em_{num_encoder_layers}l_{num_heads}h{"_posenc" if pos_enc else ""}_{str(dropout).replace(".","")}dr', organism, device=device, not_relevant=not_relevant)
    except Exception as e:
        print(e)
        print("Not found:")
        print(f'encoder_{embed_dim}em_{num_encoder_layers}l_{num_heads}h{"_posenc" if pos_enc else ""}_{str(dropout).replace(".","")}dr')
        return

    encoder_classifier = Encoder_Classifier(model)
    amino_seq = df['translation']
    true_codons = df['codons']
    pred_codons_replaced = encoder_classifier.predict_codons(amino_seq, replace=True)

    accuracy = round(encoder_classifier.calc_accuracy(true_codons, pred_codons_replaced), 4)
    print(f"Accuracy: {accuracy} - Organism: {organism}, Encoder Model - Parameters: {embed_dim} embedding dim, {num_encoder_layers} layers, {num_heads} heads")
    print(f"Took {round(time.time() - start_time, 2)} seconds")
    print("")
    return accuracy


def eval_hyperparameter_training(accuracies, embed_dims, num_encoder_layers, num_heads, dropouts, pos_enc):
    for EMBED_DIM in embed_dims:
        for NUM_ENCODER_LAYERS in num_encoder_layers:
            for NUM_HEADS in num_heads:
                for DROPOUT in dropouts:
                    for POS_ENC in pos_enc:
                        model_name = f'encoder_{EMBED_DIM}em_{NUM_ENCODER_LAYERS}l_{NUM_HEADS}h{"_posenc" if POS_ENC else ""}_{str(DROPOUT).replace(".","")}dr'
                        if model_name not in accuracies:
                            accuracy = eval_parameter_model(EMBED_DIM, NUM_ENCODER_LAYERS, NUM_HEADS, DROPOUT, POS_ENC, not_relevant=True)
                            if accuracy is not None:
                                accuracies[model_name] = accuracy
                            else:
                                accuracies[model_name] = 0
    print("------")
    print(accuracies)
    print("------")
    print(max(accuracies.items(), key=lambda item: item[1]))
    return accuracies