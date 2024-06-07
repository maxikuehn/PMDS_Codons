import numpy as np
import pandas as pd
import random
import sys

organism = pd.read_pickle('C:/Users/nilsr/Documents/HS-Mannheim/IM1/PMDS/Codons/data/E.Coli/cleanedData.pkl')

codon_sequences = pd.Series(organism['sequence'])
polypeptides = pd.Series(organism['translation'])

# longest = codon_sequences.apply(len).max()

codon_sequences = codon_sequences.apply(lambda x: [str(x[i:i+3]) for i in range(0,len(x)-3,3)])

big_dict = dict()
for sequence in codon_sequences:
    for index,codon in enumerate(sequence):
        if index not in big_dict:
            big_dict[index] = {}
        if codon not in big_dict[index]:
            big_dict[index][codon] = 0
        big_dict[index][codon] += 1
relative_big_dict = {}
for index in big_dict:
    total = sum(big_dict[index].values())
    relative_big_dict[index] = {k: v/total for k,v in big_dict[index].items()}

print(relative_big_dict[501][max(relative_big_dict[501])])

# big_dict = {}
# for sequence in codon_sequences:
#     for c in range(0,len(sequence)-3,3):
#         if (c//3,str(sequence[c:c+3])) not in big_dict:
#             big_dict[(c//3,str(sequence[c:c+3]))] = 1
#         else:
#             big_dict[(c//3,str(sequence[c:c+3]))] += 1

# def calc_cub_at_index_i(sequences,index):
#     index *= 3
#     icub = {}
#     for sequence in sequences:
#         if index+3 > len(sequence):
#             continue
#         if sequence[index:index+3] not in icub:
#             icub[sequence[index:index+3]] = 1
#         else:
#             icub[sequence[index:index+3]] += 1


#     [v/sum(icub.values())for v in icub.values()]
        


# calc_cub_at_index_i(codon_sequences,3)