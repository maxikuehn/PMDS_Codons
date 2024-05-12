from typing import Literal, Union

import pandas as pd
from Bio.Seq import Seq
import torch
import torch.nn.functional as F
from Bio.SeqRecord import SeqRecord
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*',
               '_']

aminoacids_to_integer = dict((a, i) for i, a in enumerate(amino_acids))
integer_to_aminoacids = dict((i, a) for i, a in enumerate(amino_acids))

codons = ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC', 'TGA',
          'TGG', 'CTT', 'CTC', 'CTA', 'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC',
          'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAA', 'AAG', 'AGT',
          'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG',
          'GGT', 'GGC', 'GGA', 'GGG', '___']

codons_to_integer = dict((c, i) for i, c in enumerate(codons))
integer_to_codons = dict((i, c) for i, c in enumerate(codons))


def aa_to_onehot_tensor(seq: Seq, device) -> Tensor:
    encoded_sequence = torch.as_tensor([aminoacids_to_integer[a] for a in seq]).to(device)
    one_hot_tensor = F.one_hot(encoded_sequence, num_classes=len(amino_acids))
    return one_hot_tensor.float()


def aa_to_int_tensor(seq: Seq, device) -> Tensor:
    encoded_sequence = torch.as_tensor([aminoacids_to_integer[a] for a in seq]).to(device)
    return encoded_sequence.int()


def codon_to_tensor(seq: Seq, device) -> Tensor:
    codon_seq = [seq[i:i + 3] for i in range(0, len(seq), 3)]
    tensor = torch.as_tensor([codons_to_integer[c] for c in codon_seq]).to(device)
    return tensor.float()


def codon_from_output(output: Tensor):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return codons[category_i], category_i


def filter_sequence_length(df, min_length, max_length):
    df['sequence_length'] = df["translation"].apply(len)
    if min_length is None:
        min_length = 0
    if max_length is None:
        max_length = df['sequence_length'].max()
    filtered_df: DataFrame = df[(df['sequence_length'] >= min_length) & (df['sequence_length'] <= max_length)]
    filtered_df = filtered_df.drop(columns=['sequence_length'])
    return filtered_df


def pad_sequence(seq: Union[Seq, SeqRecord], max_length, padding_pos, padding_char, seqRecord=True, padding_freq=1):
    if seqRecord:
        seq = seq.seq
    if max_length is not None:
        if len(seq) / padding_freq < max_length:
            if padding_pos == "left":
                seq = padding_char * padding_freq * (max_length - int(len(seq) / padding_freq)) + seq
            elif padding_pos == "right":
                seq = seq + padding_char * padding_freq * (max_length - int(len(seq) / padding_freq))
    return seq


organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]


class CodonDataset(Dataset):
    def __init__(self, organism: Literal["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"],
                 split: Literal["train", "test"] = "train",
                 min_length: int = None, max_length: int = None,
                 padding_pos: Literal["left", "right"] = "right",
                 one_hot_aa: bool = True,
                 data_path="../data",
                 device=torch.device("cpu")):
        self.device = device
        padding_char = "_"
        if organism not in organisms:
            raise ValueError(f"Organism '{organism}' is not in {organisms}")
        df = pd.read_pickle(f"{data_path}/{organism}/cleanedData_{split}.pkl")
        df = filter_sequence_length(df, min_length, max_length)
        df["translation"] = df["translation"].apply(pad_sequence, args=(max_length, padding_pos, padding_char))
        if one_hot_aa:
            df["translation"] = df["translation"].apply(aa_to_onehot_tensor, device=device)
        else:
            df["translation"] = df["translation"].apply(aa_to_int_tensor, device=device)
        df["sequence"] = df["sequence"].apply(pad_sequence, args=(max_length, padding_pos, padding_char, False, 3))
        df["sequence"] = df["sequence"].apply(codon_to_tensor, device=device)
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        aa_sequence = data["translation"]
        codon_sequence = data["sequence"]

        return aa_sequence, codon_sequence
