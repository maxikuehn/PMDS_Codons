from typing import Literal

import pandas as pd
from Bio.Seq import Seq
import torch
import torch.nn.functional as F
from Bio.SeqRecord import SeqRecord
from torch import Tensor
from torch.utils.data import Dataset

amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V', '*']

aminoacids_to_integer = dict((a, i) for i, a in enumerate(amino_acids))
integer_to_aminoacids = dict((i, a) for i, a in enumerate(amino_acids))

codons = ['TTT', 'TTC', 'TTA', 'TTG', 'TCT', 'TCC', 'TCA', 'TCG', 'TAT', 'TAC', 'TAA', 'TAG', 'TGT', 'TGC', 'TGA',
          'TGG', 'CTT', 'CTC', 'CTA', 'CTG', 'CCT', 'CCC', 'CCA', 'CCG', 'CAT', 'CAC', 'CAA', 'CAG', 'CGT', 'CGC',
          'CGA', 'CGG', 'ATT', 'ATC', 'ATA', 'ATG', 'ACT', 'ACC', 'ACA', 'ACG', 'AAT', 'AAC', 'AAA', 'AAG', 'AGT',
          'AGC', 'AGA', 'AGG', 'GTT', 'GTC', 'GTA', 'GTG', 'GCT', 'GCC', 'GCA', 'GCG', 'GAT', 'GAC', 'GAA', 'GAG',
          'GGT', 'GGC', 'GGA', 'GGG']

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


def convert_record_to_string(rec: SeqRecord) -> str:
    return str(rec.seq)


organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]
padding_options = ["left", "right"]


class CodonDataset(Dataset):
    def __init__(self, organism: Literal["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"],
                 padding: Literal["left", "right"] = None, padding_char: str = "0"):
        if organism not in organisms:
            raise ValueError(f"Organism '{organism}' is not in {organisms}")
        if padding is not None and padding not in padding_options:
            raise ValueError(f"Padding '{padding}' is not in {padding_options}")

        self.df: pd.DataFrame = pd.read_pickle(f"../data/{organism}/cleanedData.pkl")
        if padding is not None:
            self._max_seq_length = self.df["translation"].apply(len).max()
            translation_series = self.df["translation"].apply(convert_record_to_string)
            translation_series = translation_series.str.pad(width=0, side=padding,
                                                            fillchar=padding_char)
            self.df["translation"] = translation_series

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = self.df.iloc[idx]

        aa_sequence = data["translation"].seq
        codon_sequence = data["sequence"]

        aa_sequence = aa_to_onehot_tensor(aa_sequence)
        codon_sequence = codon_to_tensor(codon_sequence)

        return aa_sequence, codon_sequence
