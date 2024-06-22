import numpy as np
import random

from sklearn.metrics import accuracy_score

class Classifier:
    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)

    # receives a list of amino acid or codon sequences and returns a padded matrix with the amino acids (per sequence a row)
    @staticmethod
    def pad_and_convert_seq(seq, pad=''):
        max_length = max(len(s) for s in seq)
        padded_sequences = [s + [pad] * (max_length - len(s)) for s in seq]
        seq_matrix = np.array(padded_sequences)
        return seq_matrix

    # receives a list with amino acid sequences and returns a matrix with the predicted codons
    def predict_codons(self, amino_seq):
        pass

    # receives a matrix with true codons and a matrix of predicted codons and counts the number of errors
    def _count_errors(self, true_codons, pred_codons, pad=''):
        error_num = np.sum(pred_codons[pred_codons != pad] != true_codons[true_codons != pad])
        return error_num
    
    # receives a matrix with true codons and a matrix of predicted codons and counts the number of errors per amino acid
    def _count_errors_per_amino_acid(self, seq, true_codons, pred_codons):
        amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']
        amino_acid_errors = {}
        for amino_acid in amino_acids:
            acid_mask = seq == amino_acid
            amino_acid_errors[amino_acid] = {
                'total': np.sum(acid_mask),
                'errors': self._count_errors(true_codons[acid_mask], pred_codons[acid_mask])
            }
        return amino_acid_errors
    
    # calculates the error rate E = F / G (F: total number of errors, G: total number of codons)
    def calc_error_rate(self, true_codons, pred_codons, pad=''):
        true_codons = self.pad_and_convert_seq(true_codons)
        pred_codons = self.pad_and_convert_seq(pred_codons)
        error_num = self._count_errors(true_codons, pred_codons, pad=pad)
        return error_num / true_codons[true_codons != pad].size
    
    # calculates the accuracy = 1 - E
    def calc_accuracy(self, true_codons, pred_codons, pad=''):
        error_rate = self.calc_error_rate(true_codons, pred_codons, pad=pad)
        return 1 - error_rate

    def calc_accuracy_per_segment(self, true_codon_list, pred_codon_list, segment_size=10, cut_data_at=0.25):
        """
        Calculate the accuracy per segment for a given set of true and predicted codon lists.

        Parameters:
        true_codon_list (list): The list of true codons.
        pred_codon_list (list): The list of predicted codons.
        segment_size (int, optional): The size of each segment. Defaults to 10.

        Returns:
        segment_accuracies (list): The list of accuracies per segment.
        segment_elements (list): The list of number of elements per segment.
        """
        longest_seq = len(max(true_codon_list, key=len))

        # pad both sequences
        pred = self.pad_and_convert_seq(pred_codon_list, pad="")
        lab = self.pad_and_convert_seq(true_codon_list, pad="")

        # split sequences into segments
        pred = np.split(pred, np.arange(1, int((longest_seq + segment_size-1) / segment_size))*segment_size, axis=1)
        lab = np.split(lab, np.arange(1, int((longest_seq + segment_size-1) / segment_size))*segment_size, axis=1)

        # flatten segments
        pred = [p.flatten() for p in pred]
        lab = [l.flatten() for l in lab]

        # remove padding
        pred = [p[p != ""] for p in pred]
        lab = [l[l != ""] for l in lab]

        segment_accuracies = []
        segment_elements = []

        for i in range(len(pred)):
            if len(lab[i]) < len(lab[0]) * cut_data_at:
                break
            acc = accuracy_score(lab[i], pred[i])
            # print(f"Segment {i+1}: {acc}", lab[i], pred[i])
            segment_accuracies.append(acc)
            segment_elements.append(len(lab[i]))
        return segment_accuracies, segment_elements

    # calculates the error rate per amino acid
    def calc_amino_acid_error_rate(self, amino_seq, true_codons, pred_codons):
        amino_seq = self.pad_and_convert_seq(amino_seq)
        true_codons = self.pad_and_convert_seq(true_codons)
        amino_acid_errors = self._count_errors_per_amino_acid(amino_seq, true_codons, pred_codons)
        error_rates = {}
        for amino_acid in amino_acid_errors:
            error_rates[amino_acid] = amino_acid_errors[amino_acid]['errors'] / amino_acid_errors[amino_acid]['total']
        return error_rates
    
    # calculates the accuracies per amino acid
    def calc_amino_acid_accuracies(self, amino_seq, true_codons, pred_codons):
        error_rates = self.calc_amino_acid_error_rate(amino_seq, true_codons, pred_codons)
        accuracies = {}
        for amino_acid in error_rates:
            accuracies[amino_acid] = 1 - error_rates[amino_acid]
        return accuracies