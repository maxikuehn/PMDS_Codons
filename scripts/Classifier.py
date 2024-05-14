import numpy as np
import random

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
        error_num = self._count_errors(true_codons, pred_codons)
        return error_num / true_codons[true_codons != pad].size
    
    # calculates the accuracy = 1 - E
    def calc_accuracy(self, true_codons, pred_codons, pad=''):
        error_rate = self.calc_error_rate(true_codons, pred_codons, pad=pad)
        return 1 - error_rate
    
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