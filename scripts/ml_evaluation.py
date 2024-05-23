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
use flatten_for_plotting() to prepare the data for plotting:

#Example1: 

pred_codons = [[1, 2, 3], [4, 5, 6]]
label_codons = [[1, 2, 3], [64, 64, 60]]
pred_codons, label_codons = ml_evaluation.flatten_for_plotting(pred_codons, label_codons, name_codon=True, filter_pads=True, padding_value=64)
print(pred_codons, label_codons)

#Example 2: 

pred_codons = [['TTT', 'TTC', 'TTA', 'TTG'], ['TCT', 'TCC', 'TCA', 'TCG']]
label_codons = [['TTT', 'TTC', 'TTA', 'TTG'], ['TCT', 'TCC', 'TCA', 'TCG']]
pred_codons, label_codons = ml_evaluation.flatten_for_plotting(pred_codons, label_codons, name_codon=False)
print(pred_codons, label_codons)
"""

def filter_codons(codons: list, filter_value: int=64) -> list:
    return [c for c in codons if c != filter_value]

def filter_padding(predicted: list, labels: list, filter_value: int=64) -> list:
    # remove pairs where label is filter_value
    return zip(*[(pred, label) for pred, label in zip(predicted, labels) if label != filter_value])


def codon_to_name(codon_list: list) -> list:
    # translate codons to names
    return [ml_helper.codons[int(c)] for c in codon_list]

def translate_codons(codon_names: list) -> list:
    # translate the codons to amino acids
    return [str(Seq(c).translate()) for c in codon_names]


def print_unique_elements(pred_codons: list, true_codons: list) -> None:
    # get unique elements of the lists
    uniq_list1 = set(pred_codons)
    uniq_list2 = set(true_codons)
    print("predicted different amino acids: ",len(uniq_list1), "out of:", len(uniq_list2))
    if len(uniq_list1) != len(uniq_list2):
        print("predicted following amino acids: ",uniq_list1)


def flatten_for_plotting(predicted: list, labels: list, 
                         name_codon: bool = False, filter_pads=True, padding_value=64) -> list:
    """
    This function flattens the lists
    ------
    predicted: predicted labels
    labels: true labels
    name_codon: if True, the codons integers are translated to codon name strings
    filter_pads: if True, the padding value is filtered out
    padding_value: value to filter out
    ------
    returns: flattened lists
    """
    # flatten the lists
    predicted = [item for sublist in predicted for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    if filter_pads:
        predicted, labels = filter_padding(predicted, labels, padding_value)
        # back to list
        predicted = list(predicted)
        labels = list(labels)
    if name_codon:
        predicted = codon_to_name(predicted)
        labels = codon_to_name(labels)
    return predicted, labels




def flatten_dict(dict_d: dict) -> dict:
    """
    This function flattens a dictionary
    ------
    dict_d: dictionary to flatten (dict_example {'A': {'ACT': 1/4, 'ACC': 1/4, 'ACA': 1/4, 'ACG': 1/4}, ...})
    ------
    returns: flattened dictionary (dict_example {'ACT': 1/4, 'ACC': 1/4, 'ACA': 1/4, 'ACG': 1/4, ...})
    """
    # filter out codons so that amino acids are not in the dict anymore
    falttend_dict = {}
    # get codon out of dict
    for amino, codon_dict in dict_d.items():
        for k, v in codon_dict.items():
                falttend_dict[k] = v
    return falttend_dict

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



def plot_training(trainings_losses: list, trainings_accuracies: list) -> plt.Figure:
    """
    This function plots the training loss and accuracy
    ------
    trainings_losses: list with the training losses (1d list)
    trainings_accuracies: list with the training accuracies (1d list)
    ------
    returns: plot with the training loss and accuracy
    """
    plt.figure(figsize=(15, 5))
    #plt.suptitle(title, fontsize=20)

    plt.subplot(1, 2, 1)
    plt.plot(trainings_losses)
    plt.title('Training Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)

    plt.subplot(1, 2, 2)
    plt.plot(trainings_accuracies)
    plt.title('Training Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)

    return plt


def plot_confusion_matrix(labels: list, predicted: list, class_names: list, title: str,
                          cmap=plt.cm.Blues, normalize: str = 'true', sort_codons: bool = False) -> plt.Figure:
    """
    This function prints and plots the confusion matrix.
    ------
    labels: true labels (1d list over all samples)
    predicted: predicted labels (1d list over all samples)
    class_names: list with the class names
    title: title of the plot
    cmap: color map of the plot
    normalize: normalize the confusion matrix (else absolute values are shown)
    ------
    returns: plot with the confusion matrix
    """
    # sort codons by amino acids
    if sort_codons:
        labels, predicted = ml_helper.sort_codons(labels), ml_helper.sort_codons(predicted)
        class_names = ml_helper.codons_sorted

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predicted, normalize=normalize)

    plt.figure(figsize=(15,10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Vorhersage', fontsize=15)
    plt.ylabel('Richtige Kategorie', fontsize=15)
    plt.xticks(rotation=90)

    return plt


def plot_confusion_matrix_sns(labels: list, predicted: list, class_names: list,
                              title: str, cmap: str = 'coolwarm', normalize: str = 'true', sort_codons: bool = False) -> plt.Figure:
    """
    This function prints and plots the confusion matrix.
    ------
    labels: true labels (1d list over all samples)
    predicted: predicted labels (1d list over all samples)
    class_names: list with the class names
    title: title of the plot
    cmap: color map of the plot
    normalize: normalize the confusion matrix (else absolute values are shown)
    ------
    returns: plot with the confusion matrix
    """
    # sort codons by amino acids
    if sort_codons:
        labels, predicted = ml_helper.sort_codons(labels), ml_helper.sort_codons(predicted)
        class_names = ml_helper.codons_sorted

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predicted, normalize=normalize)

    mask = np.array(conf_matrix)
    # Plot the confusion matrix
    plt.figure(figsize=(15,10))
    sns.heatmap(conf_matrix, annot=mask, cmap=cmap, fmt='.2f',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=20)
    plt.xlabel('Vorhersage', fontsize=15)
    plt.ylabel('Richtige Kategorie', fontsize=15)
    return plt

def dict_aa_codon(codon=None, filter_codon=True, filter_value='___'):
    """
    This function returns a dictionary that maps each codon to its corresponding amino acid
    ------
    codon: codon to get the corresponding amino acid
    filter_codon: if True, the filter_value is not in the dictionary
    filter_value: value to filter out
    ------
    returns: dictionary that maps each codon to its corresponding amino acid
    """
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
    """
    This function plots the accuracy of each codon
    ------
    labels: true labels
    predicted: predicted labels
    title: title of the plot
    ------
    returns: plot with the accuracy of each codon
    """
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
        'Q': '#68b300',  # lindgreen
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
    plt.title(title, fontsize=20)
    #plt.xlabel('Codon')
    plt.ylabel('Akkuranz', fontsize=15)
    # rotate the x axis labels
    #plt.xticks(rotation=90)


    # Create x-axis labels with corresponding colors
    ax = plt.gca()
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(colors[i])
    # set size of tje x axis labels
    plt.xticks(fontsize=15)


    return plt




def plot_avg_aa_acc(labels, predicted, title='Druchschnittliche Codon Accuracy für jede Aminosäure'):
    """
    This function plots the average accuracy of each amino acid
    ------
    labels: true labels
    predicted: predicted labels
    title: title of the plot
    ------
    returns: plot with the average accuracy of each amino acid
    """
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

    # calculate for each amino acid the avg accuracy of the codons
    amino_acid_to_accuracy = defaultdict(list)
    for codon, accuracy in codon_accuracy.items():
        amino_acid = dict_aa_codon(codon)
        amino_acid_to_accuracy[amino_acid].append(accuracy)

    for amino_acid in amino_acid_to_accuracy:
        amino_acid_to_accuracy[amino_acid] = sum(amino_acid_to_accuracy[amino_acid]) / len(amino_acid_to_accuracy[amino_acid])
    
    # Get the keys, values, and colors as lists
    keys = list(amino_acid_to_accuracy.keys())
    values = list(amino_acid_to_accuracy.values())
    #colors = [amino_acid_to_color[dict_aa_codon(key)] for key in keys]
    color = '#219ebc'
    # plot the accuracy of each codon
    plt.figure(figsize=(20, 5))
    plt.bar(keys, values, color=color)
    plt.title(title, fontsize=20)
    plt.xlabel('Aminosäure', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    # rotate the x axis labels
    #plt.xticks(rotation=90)
    return plt

def codon_count(predicted):
    """
    This function counts the number of times each codon is predicted for each amino acid
    ------
    predicted: predicted labels
    ------
    returns: dictionary with the count of each codon
    """
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



def plot_codon_count(codon_counts, title='Anzahl Vorhersage für jedes Codon', flatten=True):
    """
    This function plots the count of each codon
    ------
    codon_counts: dictionary with the count of each codon
    title: title of the plot
    flatten: if True, the dictionary is flattened, needed if the amino acids are also in the dictionary
    """
    if flatten:
        codon_counts = flatten_dict(codon_counts)
    #print(codon_counts)
    # boxplot 
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
        'Q': '#68b300',  # lindgreen
        'R': '#800000',  # Maroon
        'S': '#aaffc3',  # Mint
        'T': '#808000',  # Olive
        'V': '#ffd8b1',  # Apricot
        'W': '#000075',  # Navy
        'Y': '#a9a9a9',  # Grey
        '*': '#000000'   # Black for stop codon
    }

    # Get the keys, values, and colors as lists
    keys = list(codon_counts.keys())
    values = list(codon_counts.values())
    colors = [amino_acid_to_color[dict_aa_codon(key)] for key in keys]

    # Sort the keys, values, and colors based on the colors
    keys, values, colors = zip(*sorted(zip(keys, values, colors), key=lambda x: x[2]))

    # plot the accuracy of each codon
    plt.figure(figsize=(20, 5))
    plt.bar(keys, values, color=colors)
    plt.title(title, fontsize=20)
    #plt.xlabel('Codon')
    plt.ylabel('Anzahl Vorhersage', fontsize=15)
    # rotate the x axis labels
    #plt.xticks(rotation=90)


    # Create x-axis labels with corresponding colors
    ax = plt.gca()
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels(keys, rotation=90)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(colors[i])
    # set size of tje x axis labels
    plt.xticks(fontsize=15)
    return plt
