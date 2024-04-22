from random import seed
from statistics import median
from numpy.random import choice


class Baseline_classificator:
    amino_dict_unweighted = {
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
    '*':{'TAA':1/3,'TAG':1/3,'TGA':1/3}}

    def __init__(self, amino_dict= amino_dict_unweighted):
        self.amino_dict = amino_dict

# Funktion, um ein zufälliges Codon für die Säure c zu erhalten. Hierzu wird de Wahrscheinlichkeit im dict verwendet.
    def get_codon(self, c):
        codon = choice(list(self.amino_dict[c].keys()), 1,
                   p=list(self.amino_dict[c].values()))
        return codon[0]

# Funktion, um Codons für eine Sequenz zu generieren
    def get_codons(self, seq: str) -> str:
        return ''.join([self.get_codon(c) for c in seq])

# Funktion, um die Übereinstimmung von Codons zwischen zwei Sequenzen zu überprüfen
# die Anzahl an richtigen Codons wird durch die Anzahl vorhandener Codons geteilt
    def check_codons(self, true_seq: str, check_seq: str):
        amount = len(true_seq) / 3 
        true_codons = 0
        for i in range(0, len(true_seq), 3):
            if true_seq[i:i + 3] == check_seq[i:i + 3]:
                true_codons += 1
        return true_codons / amount

    def check_classificator(self, seqs: list[str],trueCodons: str, seed_number: int=42):
        seed(seed_number)
        checkedV = [self.check_codons(trueCodons, self.get_codons(seq)) for seq in seqs]
        return(checkedV)

    def check_seq(self, seq:str, guessed_codons:str):
        amount = len(seq)/3
        true_codons = 0
        for j in range(0, len(seq), 3):
                if seq[j:j + 3] == guessed_codons[j:j + 3]:
                    true_codons += 1
        return[true_codons,amount]
        
    def check_classificator_new(self, df, seed_number: int=1):
        true_codons = 0
        amount = 0
        for i in df.index:
            tmp= self.check_seq(df['sequence'][i],self.get_codons(df['translation'][i].seq))
            true_codons += tmp[0]
            amount += tmp[1]
        error_rate= 1-amount/true_codons
        return {'amount': amount, 'true_codons': true_codons,'error_rate:':error_rate}

def get_highest(bias):
    max_dict = {}
    for key, sub_dict in bias.items():
        max_dict[key]={}
        for k,v in sub_dict.items():   
            if v==max(sub_dict.values()):
                max_dict[key][k]=1
    return max_dict
#import pandas as pd
#dfs = {}
#dfs['E.Coli'] = pd.read_pickle(f"data\E.Coli\cleanedData.pkl")


#classificator = Baseline_classificator()

#tmp = classificator.check_classificator_new(dfs['E.Coli'])
#print(tmp)