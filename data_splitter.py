from typing import Any
import numpy as np
import pandas as pd

def split_n_pickle(df:pd.DataFrame,name:str)->Any:
    seeds = np.array([42,212,501,187,808,999,1700,14,75,203])
    np.random.seed(seeds[np.random.randint(0,len(seeds))])
    np.random.shuffle(df)
    x1 = df[:len(df)*0.8,:]
    x2 = df[len(df)*0.8:,:]
    x1.to_pickle('Codons/data/',name,'/'+str(name)+"_train")
    x2.to_pickle('Codons/data/',name,'/'+str(name)+"_test")
    