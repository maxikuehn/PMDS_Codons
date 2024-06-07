import pandas as pd
import numpy as np
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from collections import Counter


def split_n_pickle(organism: str) -> None:
    df: pd.DataFrame = pd.read_pickle(f"./data/{organism}/cleanedData.pkl")
    seed = 42
    train_df = df.sample(frac=0.8, random_state=seed)
    test_valid_df = df.drop(train_df.index)
    test_df = test_valid_df.sample(frac=0.5, random_state=seed)
    valid_df = test_valid_df.drop(test_df.index)

    print(f"{organism:<25} total: {df.shape[0]:6} | train: {train_df.shape[0]:6} | test: {test_df.shape[0]:6} | valid: {valid_df.shape[0]:6}")

    train_df.to_pickle(f"./data/{organism}/cleanedData_train.pkl")
    test_df.to_pickle(f"./data/{organism}/cleanedData_test.pkl")
    valid_df.to_pickle(f"./data/{organism}/cleanedData_valid.pkl")

def shuffle_n_pickle(organism: str) -> None:

    data_type = ["valid", "test", "train"]
    for dt in data_type:
        df: pd.DataFrame = pd.read_pickle(f"./data/{organism}/cleanedData_{dt}.pkl")
        org_columns = df.columns.copy()
        # Shuffle index 
        df['random_index'] = df['translation'].apply(lambda x: np.random.permutation(len(x)))
        # Shuffle 'translation' sequences based on 'random_index'
        df['shuffled_translation'] = df.apply(lambda row: SeqRecord(Seq(''.join(np.array(list(str(row['translation'].seq)))[row['random_index']])), id=row['translation'].id), axis=1)
        # grouped sequences by 3 for codon shuffling
        df['seq_grouped'] = df['sequence'].apply(lambda x: [str(x)[i:i+3] for i in range(0, len(str(x)), 3)])
        # shuffle
        shuffled_seq = df.apply(lambda row: [row['seq_grouped'][i] for i in row['random_index']], axis=1)
        df['shuffeld_sequence'] = shuffled_seq
        # Concatenate the shuffled codons
        df['shuffeld_sequence'] = df.apply(lambda row: SeqRecord(Seq(''.join(row['shuffeld_sequence']))), axis=1)
 
        # Checks for data quality
        # check if shuffled codon and amino sequences are valid
        df['seq_counts'] = df['sequence'].apply(lambda x: dict(Counter(x)))
        df['shuf_seq_counts'] = df['shuffeld_sequence'].apply(lambda x: dict(Counter(x)))
        df['are_seq_dicts_same'] = df.apply(lambda row: row['seq_counts'] == row['shuf_seq_counts'], axis=1)
        not_same_count = df['are_seq_dicts_same'].value_counts().get(False, 0)
        assert not_same_count == 0, f"Codon Sequence counter doesnt match in shuffeling: {not_same_count}"

        df['back_translated_seq'] = df['shuffeld_sequence'].apply(lambda x: x.translate())
        df['seq_counts'] = df['translation'].apply(lambda x: dict(Counter(x)))
        df['shuf_seq_counts'] = df['back_translated_seq'].apply(lambda x: dict(Counter(x)))
        df['are_dicts_same'] = df.apply(lambda row: row['seq_counts'] == row['shuf_seq_counts'], axis=1)
        not_same_count = df['are_dicts_same'].value_counts().get(False, 0)
        assert not_same_count == 0, f"Amino Acid Sequence counter doesnt match in shuffeling: {not_same_count}"

        # check if shuffeld amino acids matches with translated seq
        df['is_translation_same'] = df.apply(lambda row: str(row['shuffled_translation'].seq) == str(df['back_translated_seq'][row.name].seq), axis=1)
        not_same_count = df['is_translation_same'].value_counts().get(False, 0)
        assert not_same_count == 0, f"Shuffeld Amino Acid Sequence and  Translated shuffeld codon seqeunce doesnt match in shuffeling: {not_same_count}"

        df['sequence'] = df['shuffeld_sequence']
        df['translation'] = df['shuffled_translation']

        df = df[org_columns]
        df.to_pickle(f"./data/{organism}/cleanedData_{dt}_shuffled.pkl")
        print(f'Saved file: {f"./data/{organism}/cleanedData_{dt}_shuffled.pkl"}')


organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]
for organism in organisms:
    split_n_pickle(organism)
    #shuffle_n_pickle(organism)
