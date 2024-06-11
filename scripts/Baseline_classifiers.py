from random import seed
from statistics import median
import numpy as np
import random
from numpy.random import choice
from Classifier import Classifier
import ml_helper as mlh

class Bias_Weighted_Classifier(Classifier):
    # Bias must be a dict with amino acids as keys and dicts with codons as keys and biases as values
    def __init__(self, bias, seed=42):
        self.bias = bias
        super().__init__(seed)

    def _predict_codon(self, amino_acid):
        if amino_acid == '':
            return ''
        if amino_acid == 'X':
            return 'NNN'
        probabilities = list(self.bias[amino_acid].values())
        codons = list(self.bias[amino_acid].keys())
        random_codon = random.choices(codons, weights=probabilities)[0]
        return random_codon

    def predict_codons(self, amino_seqs):
        seq_matrix = self.pad_and_convert_seq(amino_seqs)
        vectorized_func = np.vectorize(self._predict_codon)
        pred_codons_matrix = vectorized_func(seq_matrix)
        return pred_codons_matrix

class Unweighted_Baseline_Classifier(Bias_Weighted_Classifier):
    bias = {
        'A': {'GCT': 1/4, 'GCC': 1/4, 'GCA': 1/4, 'GCG': 1/4},
        'R': {'CGT': 1/6, 'CGC': 1/6, 'CGA': 1/6, 'CGG': 1/6, 'AGA': 1/6, 'AGG': 1/6},
        'N': {'AAT': 1/2, 'AAC': 1/2},
        'D': {'GAT': 1/2, 'GAC': 1/2},
        'C': {'TGT': 1/2, 'TGC': 1/2},
        'Q': {'CAA': 1/2, 'CAG': 1/2},
        'E': {'GAA': 1/2, 'GAG': 1/2},
        'G': {'GGT': 1/4, 'GGC': 1/4, 'GGA': 1/4, 'GGG': 1/4},
        'H': {'CAT': 1/2, 'CAC': 1/2},
        'I': {'ATT': 1/3, 'ATC': 1/3, 'ATA': 1/3},
        'L': {'TTA': 1/6, 'TTG': 1/6, 'CTT': 1/6, 'CTC': 1/6, 'CTA': 1/6, 'CTG': 1/6},
        'K': {'AAA': 1/2, 'AAG': 1/2},
        'M': {'ATG': 1},
        'F': {'TTT': 1/2, 'TTC': 1/2},
        'P': {'CCT': 1/4, 'CCC': 1/4, 'CCA': 1/4, 'CCG': 1/4},
        'S': {'TCT': 1/6, 'TCC': 1/6, 'TCA': 1/6, 'TCG': 1/6, 'AGT': 1/6, 'AGC': 1/6},
        'T': {'ACT': 1/4, 'ACC': 1/4, 'ACA': 1/4, 'ACG': 1/4},
        'W': {'TGG': 1},
        'Y': {'TAT': 1/2, 'TAC': 1/2},
        'V': {'GTT': 1/4, 'GTC': 1/4, 'GTA': 1/4, 'GTG': 1/4},
        '*':{'TAA':1/3,'TAG':1/3,'TGA':1/3}
    }

    def __init__(self, seed=42):
        super().__init__(self.bias, seed)

class Max_Bias_Baseline_Classifier(Bias_Weighted_Classifier):
    def __init__(self, usage_bias, seed=42):
        super().__init__(usage_bias, seed)

    def _predict_codon(self, amino_acid):
        if amino_acid == '':
            return ''
        codon_probs = self.bias[amino_acid]
        return max(codon_probs, key=codon_probs.get)
    

# Checks based on the given aa sequences and codon predictions 
# if there are codons used that do not belong to the amino acid
# and use max cub classifier for these instead
def check_and_replace_codons(aa_sequences, codon_predictions_list, usage_biases):
    max_weighted_bc = Max_Bias_Baseline_Classifier(usage_biases)
    total_codons = 0
    not_possible_codons = 0
    used_max_bias = 0
    for i, aa_seq in enumerate(aa_sequences):
        codon_predictions = codon_predictions_list[i]
        for j, aa in enumerate(aa_seq):
            total_codons += 1
            codon_pred = codon_predictions[j]
            max_bias_pred = max_weighted_bc._predict_codon(aa)
            if codon_pred not in mlh.amino_acids_to_codons[aa]:
                not_possible_codons += 1
                
                codon_predictions_list[i][j] = max_bias_pred
            else:
                if codon_pred == max_bias_pred:
                    used_max_bias += 1
    max_bias_ratio = used_max_bias / total_codons
    print(f"Model used max bias codon for {max_bias_ratio*100:.2f}% of possible codon predictions")
    not_possible_ratio = not_possible_codons / total_codons
    print(f"Replaced {not_possible_ratio*100:.2f}% of codons")
    return codon_predictions_list


# Returns the max cub accuracy for a specific organism
def get_max_cub_accuracy(organism, df, usage_biases):
    max_weighted_bc = Max_Bias_Baseline_Classifier(usage_biases)
    amino_seq = df['translation'].apply(lambda seq: list(seq))
    true_codons = df['codons']
    pred_codons_bc = max_weighted_bc.predict_codons(amino_seq)
    accuracy = max_weighted_bc.calc_accuracy(true_codons, pred_codons_bc)
    accuracy = round(accuracy, 4)
    print(f"Organismus {organism} - Accuracy: {accuracy}")