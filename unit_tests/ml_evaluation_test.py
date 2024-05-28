import os
import sys

# Get the directory that this script file is in
script_dir = os.path.dirname(os.path.realpath(__file__))
# Calculate the absolute path to the 'scripts' directory
scripts_dir = os.path.join(script_dir, '../scripts')
# Add the 'scripts' directory to the system path
sys.path.append(scripts_dir)

import ml_helper
import ml_evaluation

"""
run the following command in the terminal to run the tests:

pytest unit_tests/ml_evaluation_test.py          
"""


def test_flatten_for_plotting_filters_padding():
    # Test if padding is filtered correctly (it should be filtered if it is a label)
    pred_codons = [[1, 2, 3], [4, 5, 6]]
    label_codons = [[1, 2, 3], [64, 64, 60]]
    pred_codons, label_codons = ml_evaluation.flatten_for_plotting(pred_codons, label_codons, name_codon=False, filter_pads=True, padding_value=64)
    assert pred_codons == [1, 2, 3, 6]
    assert label_codons == [1, 2, 3, 60]

def test_flatten_for_plotting_adds_codon_names():
    # Test if codon names are added correctly
    pred_codons = [[1, 2, 3], [4, 5, 6]]
    label_codons = [[1, 2, 3], [64, 64, 60]]
    pred_codons, label_codons = ml_evaluation.flatten_for_plotting(pred_codons, label_codons, name_codon=True, filter_pads=True, padding_value=64)
    assert pred_codons == ['TTC', 'TTA', 'TTG', 'TCA']
    assert label_codons == ['TTC', 'TTA', 'TTG', 'GGT']

def test_flatten_for_plotting_without_padding_and_name_setting():
    # Test if without padding and name setting, the function works correctly
    pred_codons = [['TTT', 'TTC', 'TTA', 'TTG'], ['TCT', 'TCC', 'TCA', 'TCG']]
    label_codons = [['TTT', 'TTC', 'TTA', 'TTG'], ['TCT', 'TCC', 'TCA', 'TCG']]
    pred_codons, label_codons = ml_evaluation.flatten_for_plotting(pred_codons, label_codons, name_codon=False)
    assert pred_codons == ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG']
    assert label_codons == ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG']