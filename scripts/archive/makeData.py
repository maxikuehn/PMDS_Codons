from Baseline_classifiers import Bias_Weighted_Classifier
from Classifier import Classifier
import numpy as np
import pandas as pd
dfs={}
usage_biases={}
organisms = ["Drosophila.Melanogaster"]#"Drosophila.Melanogaster", "Homo.Sapiens"
for organism in organisms:
    dfs[organism] = pd.read_pickle(f"data/{organism}/cleanedData.pkl")
    amino_seq = dfs[organism]['translation'].apply(lambda seq: list(seq))
    usage_biases[organism] = pd.read_pickle(f"data/{organism}/usageBias.pkl")
    
    bias_classifier = Bias_Weighted_Classifier(usage_biases[organism])
    predicted_sequence = bias_classifier.predict_codons(amino_seq)
    dfs[organism]['random_sequence'] = list(predicted_sequence)

    dfs[organism].to_pickle(f"data/{organism}/cleanedData.pkl")
    print(dfs[organism].columns)
    print(dfs[organism].head()['random_sequence'])
    