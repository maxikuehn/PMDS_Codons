import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from Bio.Seq import Seq
from collections import defaultdict
import ml_helper
from mpl_toolkits.mplot3d import Axes3D

"""
This scritpt contains functions to evaluate the model and plot the results

"""

def filter_codons(codons, filter_value=64):
    return [c for c in codons if c != filter_value]

def filter_padding(predicted, labels, filter_value=64):
    # remove pairs where label is filter_value
    return zip(*[(pred, label) for pred, label in zip(predicted, labels) if label != filter_value])

def codon_to_name(codon_list):
    # translate codons to names
    return [ml_helper.codons[int(c)] for c in codon_list]

def translate_codons(codon_names):
    # translate the codons to amino acids
    return [str(Seq(c).translate()) for c in codon_names]


def print_unique_elements(list1, list2):
    uniq_list1 = set(list1)
    uniq_list2 = set(list2)

    print("predicted different amino acids: ",len(uniq_list1), "out of:", len(uniq_list2))
    if len(uniq_list1) != len(uniq_list2):
        print("predicted following amino acids: ",uniq_list1)


def get_unique_pred_classes(predicted: list, labels: list) -> set:
    """
    This function returns the unique predicted classes
    ------
    predicted: predicted labels
    labels: true labels
    ------
    returns: unique predicted classes
    """
    if torch.is_tensor(predicted):
        predicted = predicted.tolist()
    if torch.is_tensor(labels):
        labels = labels.tolist()

    unique_elements = set(predicted)
    uni_lab = set(labels)

    print("predicted different classes: ",len(unique_elements), "out of:", len(uni_lab))
    if len(unique_elements) != len(uni_lab):
        print("predicted following classes: ",unique_elements)
    return unique_elements, uni_lab

def compute_accuracy(predictions: list, labels: list) -> float:
    """
    This function computes the accuracy of the model
    -------
    predictions: list with the predicted labels
    labels: list with the true labels
    -------
    returns: accuracy of the model
    """
    acc = accuracy_score(labels, predictions)
    return acc


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
            accuracy = compute_accuracy(predicted.cpu(), labels.cpu())
            accuracies.append(accuracy)

    # Compute average accuracy
    
    return predicted, labels, accuracies



def plot_training(trainings_losses, trainings_accuracies, title='Training Loss and Accuracy'):
    plt.figure(figsize=(15, 5))
    plt.suptitle(title)

    plt.subplot(1, 2, 1)
    plt.plot(trainings_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(trainings_accuracies)
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    return plt


def plot_confusion_matrix(labels, predicted, classes, title, cmap=plt.cm.Blues, normalize='true'):
    """
    This function prints and plots the confusion matrix.
    """
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predicted, normalize=normalize)

    plt.figure(figsize=(15,10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    plt.xlabel('Vorhersage')
    plt.ylabel('Richtige Kategorie')
    return plt

def plot_confusion_matrix_sns(labels, predicted, classes, title, cmap='coolwarm', normalize='true'):
    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predicted, normalize=normalize)

    mask = np.array(conf_matrix)
    # Plot the confusion matrix
    plt.figure(figsize=(15,10))
    sns.heatmap(conf_matrix, annot=mask, cmap=cmap, fmt='.2f',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.xlabel('Vorhersage')
    plt.ylabel('Richtige Kategorie')
    return plt

def dict_aa_codon(codon=None, filter_codon=True, filter_value='___'):
    all_codons = ml_helper.codons
    if filter_codon:
        all_codons = [c for c in all_codons if c != filter_value]
    # Create a dictionary that maps each codon to its corresponding amino acid
    codon_to_amino_acid = {codon: str(Seq(codon).translate()) for codon in all_codons}
    if codon is None:
        return codon_to_amino_acid
    else:
        return codon_to_amino_acid[codon]
    

def plot_codon_acc(labels, predicted, title='Akkuranz für jedes Codon'):
    labels_codon_names = codon_to_name(labels)
    predicted_codon_names = codon_to_name(predicted)

    # calculate for each codon the accuaracy
    codon_accuracy = {}

    for i in range(len(labels_codon_names)):
        if labels_codon_names[i] not in codon_accuracy:
            codon_accuracy[predicted_codon_names[i]] = 0
        if predicted_codon_names[i] == labels_codon_names[i]:
            codon_accuracy[predicted_codon_names[i]] += 1

    for key in codon_accuracy:
        codon_accuracy[key] = codon_accuracy[key] / len([c for c in labels_codon_names if c == key])


    amino_acid_to_color = {
        'A': '#e6194B',  # Red
        'C': '#3cb44b',  # Green
        'D': '#ffe119',  # Yellow
        'E': '#4363d8',  # Blue
        'F': '#f58231',  # Orange
        'G': '#911eb4',  # Purple
        'H': '#42d4f4',  # Cyan
        'I': '#f032e6',  # Magenta
        'K': '#bfef45',  # Lime
        'L': '#fabed4',  # Pink
        'M': '#469990',  # Teal
        'N': '#dcbeff',  # Lavender
        'P': '#9A6324',  # Brown
        'Q': '#fffac8',  # Beige
        'R': '#800000',  # Maroon
        'S': '#aaffc3',  # Mint
        'T': '#808000',  # Olive
        'V': '#ffd8b1',  # Apricot
        'W': '#000075',  # Navy
        'Y': '#a9a9a9',  # Grey
        '*': '#000000'   # Black for stop codon
    }

    # Get the keys, values, and colors as lists
    keys = list(codon_accuracy.keys())
    values = list(codon_accuracy.values())
    colors = [amino_acid_to_color[dict_aa_codon(key)] for key in keys]

    # Sort the keys, values, and colors based on the colors
    keys, values, colors = zip(*sorted(zip(keys, values, colors), key=lambda x: x[2]))

    # plot the accuracy of each codon
    plt.figure(figsize=(20, 5))
    plt.bar(keys, values, color=colors)
    plt.title(title)
    plt.xlabel('Codon')
    plt.ylabel('Akkuranz')
    # rotate the x axis labels
    plt.xticks(rotation=90)
    return 


def codon_count(predicted):
    predicted_codon_names = codon_to_name(predicted)
    # Initialize a dictionary to count the number of times each codon is predicted for each amino acid
    codon_counts = defaultdict(lambda: defaultdict(int))

    dict_codon = dict_aa_codon()
    # Iterate over the predicted codons
    for codon in predicted_codon_names:
        # Get the amino acid that the codon codes for
        amino_acid = dict_codon[codon]
        
        # Increment the count for this codon and amino acid
        codon_counts[amino_acid][codon] += 1

    # Create a dictionary mapping amino acids to codons
    amino_acid_to_codons = defaultdict(list)
    for codon, amino_acid in dict_codon.items():
        amino_acid_to_codons[amino_acid].append(codon)

    # Iterate over all possible codons for each amino acid
    for amino_acid, codons in amino_acid_to_codons.items():
        for codon in codons:
            # Add the codon to the dictionary if it isn't already present
            if codon not in codon_counts[amino_acid]:
                codon_counts[amino_acid][codon] = 0
    
    return codon_counts

def plot_codon_count_3d(codon_counts):
    dict_codon = dict_aa_codon()

    #for amino_acid in codon_counts.keys():
    #    if amino_acid not in dict_codon:
    #        dict_codon[amino_acid] = 'b'

    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Iterate over the amino acids
    for i, (amino_acid, counts) in enumerate(codon_counts.items()):
        # Get the codons and counts
        codons = list(counts.keys())
        count_values = list(counts.values())
        
        # Create a list of the x, y, and z coordinates
        x = [i] * len(codons)
        y = list(range(len(codons)))
        z = [0] * len(codons)
        
        # Create a bar for each codon
        if amino_acid in dict_codon:  # Check if the amino acid is in dict_codon
            ax.bar3d(x, y, z, 1, 1, count_values, color=dict_codon[amino_acid])


    # Set the x, y, and z labels
    ax.set_xlabel('Amino Acid')
    ax.set_ylabel('Codon')
    ax.set_zlabel('Count')

    return plt