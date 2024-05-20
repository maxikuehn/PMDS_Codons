import os
import datetime
from typing import Literal, Union

import pandas as pd
from Bio.Seq import Seq
import torch
import torch.nn as nn
import torch.nn.functional as F
from Bio.SeqRecord import SeqRecord
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*',
               '_']

amino_acids_to_codons = {
    'A': ['GCT', 'GCC', 'GCA', 'GCG'],
    'R': ['CGT', 'CGC', 'CGA', 'CGG', 'AGA', 'AGG'],
    'N': ['AAT', 'AAC'],
    'D': ['GAT', 'GAC'],
    'C': ['TGT', 'TGC'],
    'Q': ['CAA', 'CAG'],
    'E': ['GAA', 'GAG'],
    'G': ['GGT', 'GGC', 'GGA', 'GGG'],
    'H': ['CAT', 'CAC'],
    'I': ['ATT', 'ATC', 'ATA'],
    'L': ['TTA', 'TTG', 'CTT', 'CTC', 'CTA', 'CTG'],
    'K': ['AAA', 'AAG'],
    'M': ['ATG'],
    'F': ['TTT', 'TTC'],
    'P': ['CCT', 'CCC', 'CCA', 'CCG'],
    'S': ['TCT', 'TCC', 'TCA', 'TCG', 'AGT', 'AGC'],
    'T': ['ACT', 'ACC', 'ACA', 'ACG'],
    'W': ['TGG'],
    'Y': ['TAT', 'TAC'],
    'V': ['GTT', 'GTC', 'GTA', 'GTG'],
    '*': ['TAA', 'TAG', 'TGA']
}

aminoacids_to_integer = dict((a, i) for i, a in enumerate(amino_acids))
integer_to_aminoacids = dict((i, a) for i, a in enumerate(amino_acids))

codons = ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC', 'TGA',
          'TGG', 'CTT', 'CTC', 'CTA', 'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC',
          'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAA', 'AAG', 'AGT',
          'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG',
          'GGT', 'GGC', 'GGA', 'GGG', '___']

codons_sorted = ["TTT", "TTC", "TTA", "TTG", "CTT", "CTC", "CTA", "CTG", "ATT", "ATC", "ATA", "ATG", "GTT", "GTC",
                 "GTA", "GTG", "TCT", "TCC", "TCA", "TCG", "AGT", "AGC", "CCT", "CCC", "CCA", "CCG", "ACT", "ACC", "ACA", "ACG",
                 "GCT", "GCC", "GCA", "GCG", "TAT", "TAC", "CAT", "CAC", "CAA", "CAG", "AAT", "AAC",
                 "AAA", "AAG", "GAT", "GAC", "GAA", "GAG", "TGT", "TGC", "TGG", "CGT", "CGC", "CGA", "CGG",
                 "AGA", "AGG", "GGT", "GGC", "GGA", "GGG", "TAA", "TAG", "TGA"]

codons_to_integer = dict((c, i) for i, c in enumerate(codons))
integer_to_codons = dict((i, c) for i, c in enumerate(codons))
codons_to_sorted_integer = dict((c, i) for i, c in enumerate(codons_sorted))
integer_to_sorted_codons = dict((i, c) for i, c in enumerate(codons_sorted))


def aa_int_to_onehot_tensor(tensor: Tensor) -> Tensor:
    tensor = tensor.long()
    one_hot_tensor = F.one_hot(tensor, num_classes=len(amino_acids))
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


def sort_codons(codons):
    codon_names = [integer_to_codons[i] for i in codons]
    sorted_integers = [codons_to_sorted_integer[c] for c in codon_names]
    return sorted_integers


def filter_sequence_length(df, min_length, max_length):
    df['sequence_length'] = df["translation"].apply(len)
    if min_length is None:
        min_length = 0
    if max_length is None:
        max_length = df['sequence_length'].max()
    filtered_df: DataFrame = df[(df['sequence_length'] >= min_length) & (df['sequence_length'] <= max_length)]
    filtered_df = filtered_df.drop(columns=['sequence_length'])
    return filtered_df


# Function to split a tensor into chunks of max_length characters
def _split_tensor(t, max_length=500):
    chunk_tensors = [t[i:i+max_length] for i in range(0, len(t), max_length)]
    return chunk_tensors


def cut_sequences(df, max_length):
    new_df = pd.DataFrame(columns=["translation", "sequence"])
    new_rows = []
    for _, row in df.iterrows():
        aa_sequence = row["translation"]
        codon_sequence = row["sequence"]
        if aa_sequence.shape[0] <= max_length:
            new_rows.append({"translation": row["translation"], "sequence": row["sequence"]})
        elif aa_sequence.shape[0] > max_length:
            aa_splits = _split_tensor(aa_sequence, max_length)
            codon_splits = _split_tensor(codon_sequence, max_length)
            for i, aa_split in enumerate(aa_splits):
                new_rows.append({"translation": aa_split, "sequence": codon_splits[i]})
    new_df = pd.DataFrame(new_rows)
    return new_df


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


def pad_tensor(tensor: Tensor, max_length, padding_symbol, padding_pos):
    if padding_pos == "left":
        return F.pad(tensor, (max_length - len(tensor), 0), value=padding_symbol)
    elif padding_pos == "right":
        return F.pad(tensor, (0, max_length - len(tensor)), value=padding_symbol)


organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]


class CodonDataset(Dataset):
    def __init__(self,
                 organism: Literal["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"],
                 split: Literal["train", "test"] = "train",
                 min_length: int = None,
                 max_length: int = None,
                 cut_data: bool = False,
                 padding_pos: Literal["left", "right"] = "right",
                 one_hot_aa: bool = True,
                 data_path="../data",
                 device=torch.device("cpu")):

        if organism not in organisms:
            raise ValueError(f"Organism '{organism}' is not in {organisms}")
        if cut_data and max_length == None:
            raise ValueError(f"cut_data=True needs a given max_length to cut")

        self.organism = organism
        self.split = split
        self.min_length = min_length
        self.max_length = max_length
        self.cut_data = cut_data
        self.padding_pos = padding_pos
        self.one_hot_aa = one_hot_aa
        self.device = device
        self.padding_char = "_"

        df = pd.read_pickle(f"{data_path}/{organism}/cleanedData_{split}.pkl")

        if not cut_data:
            df = filter_sequence_length(df, min_length, max_length)

        if padding_pos:
            df["translation"] = df["translation"].apply(pad_sequence, args=(max_length, padding_pos, self.padding_char))
            df["sequence"] = df["sequence"].apply(pad_sequence, args=(max_length, padding_pos, self.padding_char, False, 3))

        df["translation"] = df["translation"].apply(aa_to_int_tensor, device=device)
        df["sequence"] = df["sequence"].apply(codon_to_tensor, device=device)

        if cut_data:
            df = cut_sequences(df, max_length)
            df["translation"] = df["translation"].apply(pad_tensor, args=(max_length, aminoacids_to_integer['_'], padding_pos))
            df["sequence"] = df["sequence"].apply(pad_tensor, args=(max_length, codons_to_integer["___"], padding_pos))

        if one_hot_aa:
            df["translation"] = df["translation"].apply(aa_int_to_onehot_tensor)

        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        aa_sequence = data["translation"]
        codon_sequence = data["sequence"]

        return aa_sequence, codon_sequence


def save_model(model: nn.Module,  model_name: str, organism: str, appendix: str = None):
    # timestamp
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d%H%M%S")

    appendix = "_" + appendix if appendix is not None else ""

    path = f"../ml_models/{organism}/{timestamp}_{model_name}{appendix}.pt"

    # save model in ml_models in a single file
    torch.save(model, path)
    print(f"Model saved as {timestamp}_{model_name}{appendix}.pt")


def load_model(model_name: str, organism: str):
    # get the newest version of the tcn model
    organism_models = os.listdir(f"../ml_models/{organism}")
    # get all models from type
    models = [model for model in organism_models if model_name in model]
    # sort by date
    models.sort()
    # get newest model
    newest_model = models[-1]
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(f"../ml_models/{organism}/{newest_model}", map_location=device)
    print(f"Model loaded: {newest_model}")
    return model
