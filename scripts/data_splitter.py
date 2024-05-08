from typing import Any
import numpy as np
import pandas as pd

def split_n_pickle(organism:str)->Any:
    df = pd.read_pickle(f"../data/{organism}/cleanedData.pkl")
    seed = 42
    train_df = df.sample(frac=0.8, random_state=seed)
    test_df = df.drop(train_df.index)
    train_df.to_pickle('../data/'+organism+'/'+"cleanedData_train.pkl")
    test_df.to_pickle('../data/'+organism+'/'+"cleanedData_test.pkl")

organisms = ["E.Coli", "Drosophila.Melanogaster", "Homo.Sapiens"]
for organism in organisms:
    split_n_pickle(organism)
    