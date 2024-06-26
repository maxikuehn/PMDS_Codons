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
import pandas as pd
import Baseline_classifiers as bc

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

def plot_training(trainings_losses: list, trainings_accuracies: list, valid_accs: list = None) -> plt.Figure:
    """
    This function plots the training loss and accuracy.
    ------
    trainings_losses: list with the training losses (1d list)
    trainings_accuracies: list with the training accuracies (1d list)
    valid_accs: list with the validation accuracies (1d list), optional
    ------
    returns: plot with the training loss and accuracy
    """
    plt.figure(figsize=(15, 5))

    # Plot training loss
    plt.subplot(1, 2, 1)
    plt.plot(trainings_losses, label='Training Loss')
    plt.title('Training Loss', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.legend()

    # Plot training and validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(trainings_accuracies, label='Training Accuracy')
    if valid_accs is not None:
        plt.plot(valid_accs, label='Validation Accuracy')
    plt.title('Training Accuracy', fontsize=20)
    plt.xlabel('Epoch', fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.legend()

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
        class_names = ml_helper.codons_sorted_no_stop

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predicted, normalize=normalize)

    plt.figure(figsize=(15,10))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Correct Category', fontsize=15)
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
        class_names = ml_helper.codons_sorted_no_stop

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(labels, predicted, normalize=normalize)

    mask = np.array(conf_matrix)
    # Plot the confusion matrix
    plt.figure(figsize=(15,10))
    sns.heatmap(conf_matrix, annot=mask, cmap=cmap, fmt='.2f',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title, fontsize=20)
    plt.xlabel('Prediction', fontsize=15)
    plt.ylabel('Correct Category', fontsize=15)
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
    

def plot_codon_acc(labels, predicted, title='Accuracy für jedes Codon'):
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
    codon_correct = {}

    for i in range(len(labels_codon_names)):
        if labels_codon_names[i] not in codon_correct:
            codon_correct[labels_codon_names[i]] = 0
        if predicted_codon_names[i] == labels_codon_names[i]:
            codon_correct[labels_codon_names[i]] += 1

    codon_accuracies = {}
    for key in codon_correct:
        codon_accuracies[key] = codon_correct[key] / len([c for c in labels_codon_names if c == key])


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
    keys = list(codon_accuracies.keys())
    values = list(codon_accuracies.values())
    colors = [amino_acid_to_color[dict_aa_codon(key)] for key in keys]

    # Sort the keys, values, and colors based on the colors
    keys, values, colors = zip(*sorted(zip(keys, values, colors), key=lambda x: x[2]))

    # plot the accuracy of each codon
    plt.figure(figsize=(20, 5))
    plt.bar(keys, values, color=colors)
    plt.title(title, fontsize=20)
    #plt.xlabel('Codon')
    plt.ylabel('Accuracy', fontsize=15)
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


    return plt, codon_accuracies




def plot_avg_aa_acc(labels, predicted, title='Durchschnittliche Codon Accuracy für jede Aminosäure'):
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
            codon_accuracy[labels_codon_names[i]] = 0
        if predicted_codon_names[i] == labels_codon_names[i]:
            codon_accuracy[labels_codon_names[i]] += 1

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

def codon_count(predicted, labels=None):
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
    if labels is not None:
        for codon in labels:
            codon_counts[dict_codon[codon]][codon] = 0

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
    return plt, 


def plot_relative_codon_count(codon_counts, predicted, title='Relativer Anteil der Vorhersagen für jedes Codon', flatten=True):
    """
    This function plots the relative number of each codon 
    (model suggested codon usage bias)
    ------
    codon_counts: dictionary with the count of each codon
    precited: predicted labels
    title: title of the plot
    flatten: if True, the dictionary is flattened, needed if the amino acids are also in the dictionary
    """
    if flatten:
        codon_counts = flatten_dict(codon_counts)

    predicted_codon_names = codon_to_name(predicted)

    codon_to_aa = dict_aa_codon()
    relative_codon_usage = {}
    for codon in codon_counts:
        if len([key for key in predicted_codon_names if codon_to_aa[key] == codon_to_aa[codon]]) > 0:
            relative_codon_usage[codon] = codon_counts[codon] / len([key for key in predicted_codon_names if codon_to_aa[key] == codon_to_aa[codon]])

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
    keys = list(relative_codon_usage.keys())
    values = list(relative_codon_usage.values())
    colors = [amino_acid_to_color[dict_aa_codon(key)] for key in keys]

    # Sort the keys, values, and colors based on the colors
    keys, values, colors = zip(*sorted(zip(keys, values, colors), key=lambda x: x[2]))

    # plot the accuracy of each codon
    plt.figure(figsize=(20, 5))
    plt.bar(keys, values, color=colors)
    plt.title(title, fontsize=20)
    #plt.xlabel('Codon')
    plt.ylabel('Relative frequency of predictions', fontsize=15)
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


def plot_cub(cub_dict, title="Codon Usage Bias für jedes Codon"):
    """
    This function plots the Codon Usage Bias in a similar way
    as the plot_relative_codon_count function to compare the two
    """
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
    keys = list(cub_dict.keys())
    values = list(cub_dict.values())
    colors = [amino_acid_to_color[dict_aa_codon(key)] for key in keys]

    # Sort the keys, values, and colors based on the colors
    keys, values, colors = zip(*sorted(zip(keys, values, colors), key=lambda x: x[2]))

    # plot the accuracy of each codon
    plt.figure(figsize=(20, 5))
    plt.bar(keys, values, color=colors)
    plt.title(title, fontsize=20)
    #plt.xlabel('Codon')
    plt.ylabel('Relative frequency', fontsize=15)
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
    return plt, keys


def group_codons(sequence):
        return [''.join(sequence[i:i+3]) for i in range(0, len(sequence), 3)]


def max_cub_predictions(organism):
    '''
    This function returns the max cub predictions for the given organism
    as list of lists (codons for each df row)
    '''
    df = pd.read_pickle(f"../data/{organism}/cleanedData_test.pkl")
    usage_biases = pd.read_pickle(f"../data/{organism}/usageBias.pkl")
    df['codons'] = df['sequence'].apply(group_codons)

    max_weighted_bc = bc.Max_Bias_Baseline_Classifier(usage_biases)
    amino_seq = df['translation'].apply(lambda seq: list(seq))
    pred_codons_bc = max_weighted_bc.predict_codons(amino_seq)
    return pred_codons_bc


def create_pn_dict(predicted_m, labels, organism, sorting={}):
    '''
    This function creates a dictionary of the following form:
    {
        'ATG': {
            "num": 0,    # number of occurences in the labels
            "P_M==B": 0, # correct model prediction, where model is baseline
            "P_M!=B": 0, # correct model prediction, where model is not baseline
            "N_M==B": 0, # false model prediction, where model is baseline
            "N_M!=B": 0  # false model prediction, where model is not baseline
        }, ...
    }
    -----------
    predicted_m: predicted codons als string as one list (all rows concatenated)
    labels: true codons as string as one list (all rows concatenated)
    organism: organism to evaluate (important for baseline classifier)
    '''
    pred_codons_bc = max_cub_predictions(organism)
    predicted_bc = np.array(pred_codons_bc[pred_codons_bc != ''])
    predicted_m = np.array(predicted_m)
    labels = np.array(labels)

    pn_dict = {}
    sorting_dict = ml_helper.codons_sorted
    if sorting != {}:
        sorting_dict = sorting
    for codon in sorting_dict:
        if (labels == codon).sum() != 0:
            pn_dict[codon] = {
                "num": (labels == codon).sum(),
                "P_M==B": 0, # positive, where model is baseline
                "P_M!=B": 0, # positive, where model is not baseline
                "N_M==B": 0, # negative, where model is baseline
                "N_M!=B": 0  # negative, where model is not baseline
            }

    for i, codon_l in enumerate(labels):
        if codon_l == predicted_m[i]:
            if predicted_m[i] == predicted_bc[i]:
                pn_dict[codon_l]["P_M==B"] += 1 / pn_dict[codon_l]["num"]
            else:
                pn_dict[codon_l]["P_M!=B"] += 1 / pn_dict[codon_l]["num"]
        else:
            if predicted_m[i] == predicted_bc[i]:
                pn_dict[codon_l]["N_M==B"] += 1 / pn_dict[codon_l]["num"]
            else:
                pn_dict[codon_l]["N_M!=B"] += 1 / pn_dict[codon_l]["num"]
    
    return pn_dict


def plot_pn_dict(pn_dict, model_name, organism_name):
    '''
    This function plots the dictionary 
    -----------
    pn_dict: result of create_pn_dict function
    model_name: name of the model (e.g. 'Transformer')
    organism: organism name (e.g. 'Mensch')
    '''
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

    # Extract data for plotting
    labels = list(pn_dict.keys())
    colors = [amino_acid_to_color[dict_aa_codon(label)] for label in labels]
    labels, colors = zip(*sorted(zip(labels, colors), key=lambda x: x[1]))
    P_M_equal_B = [pn_dict[label]['P_M==B'] for label in labels]
    P_M_not_equal_B = [pn_dict[label]['P_M!=B'] for label in labels]
    N_M_equal_B = [pn_dict[label]['N_M==B'] for label in labels]
    N_M_not_equal_B = [pn_dict[label]['N_M!=B'] for label in labels]


    # Plotting the stacked bar chart
    plt.figure(figsize=(12, 4))

    # Define the positions of the bars
    r = np.arange(len(labels))

    # Plot each segment of the bar
    plt.bar(r, P_M_equal_B, color='darkgreen', edgecolor='grey', label='P_M==B')
    plt.bar(r, P_M_not_equal_B, bottom=P_M_equal_B, color='limegreen', edgecolor='grey', label='P_M!=B')
    plt.bar(r, N_M_equal_B, bottom=np.array(P_M_equal_B) + np.array(P_M_not_equal_B), color='darkred', edgecolor='grey', label='N_M==B')
    plt.bar(r, N_M_not_equal_B, bottom=np.array(P_M_equal_B) + np.array(P_M_not_equal_B) + np.array(N_M_equal_B), color='lightcoral', edgecolor='grey', label='N_M!=B')

    # Add labels
    plt.xlabel('Codons', fontweight='bold')
    plt.ylabel('Relative frequency', fontweight='bold')
    #plt.xticks(r, labels, rotation=45)
    plt.title(f'Frequencies of correct (P) and false (N) predictions\of {model_name} model (M) in comparison to the Max CUB baseline (B) for organism {organism_name}')

    # Add a legend
    plt.legend()
    plt.tight_layout()

    ax = plt.gca()
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90)
    for i, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(colors[i])
    # set size of tje x axis labels
    plt.xticks(fontsize=15)

    return plt


def plot_accuracies_comparison(accuracies, bar_labels, title, value_decimals=3):
    '''
    This function plots the accuracies of different organisms for each classifier
    The accuracies must be given in the following format:
    accuracies = {
        "E.Coli": {
            "Max CUB": 0.5186,
            "Transformer": 0.5264
    }, ...
    -------------------------
    accuracies: accuracies for each classifier
    bar_labels: names for the classifiers in the bar plot
    title: title for the plot
    value_decimals: on which number of decimals to round the value texts in the plot
}
    '''
    colors = ['#011f4b', '#6497b1', '#03396c', '#b3cde0', '#005b96']

    # Prepare data for plotting
    organisms = list(accuracies.keys())
    classifier_labels = accuracies[organisms[0]].keys()
    values_list = []
    for label in classifier_labels:
        labels_list = []
        for org in organisms:
            if label in accuracies[org]:
                labels_list.append(accuracies[org][label])
            else:
                labels_list.append(0)
        values_list.append(labels_list)

    # Number of bars
    x = np.arange(len(organisms))

    # Create the plot
    plot_length = 6
    if len(bar_labels) > 3:
        plot_length = 9
    fig, ax = plt.subplots(figsize=(plot_length, 4))

    # Plotting the bars
    bars = []
    ylim=(0, 1)
    bar_width = 0.4 / len(classifier_labels) * 2
    for i, values in enumerate(values_list):
        bars.append(ax.bar(x + i * bar_width - (len(values_list) - 1) * bar_width / 2, values, bar_width, label=bar_labels[i], color=colors[i]))

    # Adding labels and title
    ax.set_xlabel("Organismus")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(organisms)
    ax.set_ylim(*ylim)
    ax.legend()

    # Adding value labels on top of the bars
    def add_value_labels(bars):
        for bar_group in bars:
            for bar in bar_group:
                height = bar.get_height()
                if height != 0:
                    ax.text(bar.get_x() + bar.get_width() / 2.0, height, f'{round(height, value_decimals)}', ha='center', va='bottom')

    add_value_labels(bars)

    # Display the plot
    plt.show()


def plot_accuracies_per_segment(accuracies, elements, title):
    # Create the plot
    fig, ax1 = plt.subplots(figsize=(15, 4))
    ax1.set_title(title)

    ax1.set_xlabel("Segment")
    ax1.set_ylim(0, elements[0] * 1.05)
    ax1.set_xlim(-1, len(elements))
    ax1.set_ylabel("Number of elements per segment")
    ax1.bar(range(len(elements)), elements)

    ax2 = ax1.twinx()

    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.plot(accuracies, color='darkgreen', linewidth=3)

    plt.show()


# Function to plot training accuracies for given data
def plot_training_accuracies(training_accuracies, model_name, epoch_distance=1):
    '''
    Accuracies dict needs to be of following form:
    training_accuracies = {
        "E.Coli": [0.5213, 0.5473, 0.5563, ...],
        "Fruchtfliege": ...,
        "Mensch": ...
    }
    '''
    colors = {
        "E.Coli": 'green',
        "Fruchtfliege": 'blue',
        "Mensch": 'red'
    }

    # Plotting the training accuracies for each organism
    plt.figure(figsize=(16, 5))

    for organism, accuracies in training_accuracies.items():
        epochs = range(1, len(accuracies) + 1)
        plt.plot(epochs, accuracies, label=organism, color=colors[organism])

    # Adding labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracies on validation data over epochs for the best {model_name} model per organism')
    plt.legend()

    max_epochs = max(len(acc) for acc in training_accuracies.values())
    plt.xticks(range(0, max_epochs + 1, epoch_distance))

    # Display the plot
    plt.show()