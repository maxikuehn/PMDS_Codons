from typing import Literal

import pandas as pd
from Bio.Seq import Seq
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import Dataset

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*','_']

aminoacids_to_integer = dict((a, i) for i, a in enumerate(amino_acids))
integer_to_aminoacids = dict((i, a) for i, a in enumerate(amino_acids))

codons = ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC', 'TGA',
          'TGG', 'CTT', 'CTC', 'CTA', 'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC',
          'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAA', 'AAG', 'AGT',
          'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG',
          'GGT', 'GGC', 'GGA', 'GGG', '___']

codons_to_integer = dict((c, i) for i, c in enumerate(codons))
integer_to_codons = dict((i, c) for i, c in enumerate(codons))


def aa_to_onehot_tensor(seq: Seq) -> Tensor:
    encoded_sequence = torch.as_tensor([aminoacids_to_integer[a] for a in seq])
    one_hot_tensor = F.one_hot(encoded_sequence, num_classes=len(amino_acids))
    return one_hot_tensor.float()


def codon_to_tensor(seq: Seq) -> Tensor:
    codon_seq = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    tensor = torch.as_tensor([codons_to_integer[c] for c in codon_seq])
    return tensor.float()

def codon_from_output(output: Tensor):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return codons[category_i], category_i

def filter_sequence_length(df, min_length, max_length):
    df['sequence_length'] = df["translation"].apply(len)
    if min_length == None:
        min_length = 0
    if max_length == None:
        max_length = df['sequence_length'].max()
    filtered_df = df[(df['sequence_length'] >= min_length) & (df['sequence_length'] <= max_length)]
    filtered_df.drop(columns=['sequence_length'], inplace=True)
    return filtered_df

def pad_sequence(seq, max_length, padding_pos, padding_char, seqRecord=True, padding_freq=1):
    if seqRecord:
        seq = seq.seq
    if max_length != None:
        if len(seq)/padding_freq < max_length:
            if padding_pos == "left":
                seq = padding_char * padding_freq * (max_length - int(len(seq)/padding_freq)) + seq
            elif padding_pos == "right":
                seq = seq + padding_char * padding_freq * (max_length - int(len(seq)/padding_freq))
    return seq

organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]


class CodonDataset(Dataset):
    def __init__(self, organism: Literal["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"],
                 min_length: int = None, max_length: int = None,
                 padding_pos: Literal["left", "right"] = "right", padding_char: str = "_"):
        if organism not in organisms:
            raise ValueError(f"Organism '{organism}' is not in {organisms}")
        df = pd.read_pickle(f"../data/{organism}/cleanedData.pkl")
        df = filter_sequence_length(df, min_length, max_length)
        df["translation"] = df["translation"].apply(pad_sequence, args=(max_length, padding_pos, padding_char))
        df["translation"] = df["translation"].apply(aa_to_onehot_tensor)
        df["sequence"] = df["sequence"].apply(pad_sequence, args=(max_length, padding_pos, padding_char, False, 3))
        df["sequence"] = df["sequence"].apply(codon_to_tensor)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        aa_sequence = data["translation"]
        codon_sequence = data["sequence"]

        return aa_sequence, codon_sequence
