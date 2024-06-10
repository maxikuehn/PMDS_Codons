import pandas as pd
import numpy as np
#1
def read_organism(organism_name):
    global organism
    organism = pd.read_pickle(f"C:/Users/nilsr/Documents/HS-Mannheim/IM1/PMDS/Codons/data/{organism_name}/cleanedData.pkl")
#2
def get_codons_and_polypeptides()->tuple[pd.Series,pd.Series]:
    return pd.Series(organism['sequence']),pd.Series(organism['translation'])
#4
def mu_sigma_interval_length():
    mu = np.mean(organism['sequence'].apply(len)//3)//1
    sigma = np.std(organism['sequence'].apply(len)//3)//1
    return mu,sigma
#3
def preprocess_series(codons:pd.Series,polypeptide:pd.Series,min=0)->tuple[pd.Series,pd.Series]:    
    mu,sigma = mu_sigma_interval_length()
    upper = mu+sigma
    #polypeptides_sample = polypeptide[polypeptide.apply(len).between(0,max(polypeptide.apply(len)),'both')]
    polypeptides_sample = polypeptide.apply(lambda x: x.seq)
    polypeptides_sample = polypeptides_sample.reset_index(drop=True)
    codon_sequences = codons.apply(lambda x: [str(x[i:i+3]) for i in range(0,len(x)-3,3)])
    return codon_sequences,polypeptides_sample

"""
big_dict = {0:{'ATG':12},1:{'GCG':3,'GGG':1,...},...}
"""
#5
def calc_relative_index_bias(processed_codons:pd.Series):
    big_dict = dict()
    for sequence in processed_codons:
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
    return relative_big_dict

def max_bias_per_index(relative_big_dict):
    max_values = [max(relative_big_dict[index].items()) for index in relative_big_dict]
    filtered_max_values = []
    for element in max_values:
        if type(element) == list:
            filtered_max_values.append(element[0])
        else:
            filtered_max_values.append(element)
    return filtered_max_values

def predict_on_index(polypeptide,max_on_index):
    predicted_sequence = []
    for x in range(len(polypeptide)-1):
        predicted_sequence.append(max_on_index[x][0])
    return predicted_sequence

def predict_organism(codons,polypeptide,max_on_index):
        predicted_sequences = [predict_on_index(element,max_on_index) for element in polypeptide]
        return calc_prediction_accuraccy(predicted_sequences,codons)

def compare_prediction_with_reality(pred,real):
    total = len(real)
    correct = 0
    for i in range(len(pred)):
        if pred[i] == real[i]:
            correct+=1
    return correct/total
    
def calc_prediction_accuraccy(preds,reals):
    return np.mean([compare_prediction_with_reality(element[0],element[1]) for element in zip(preds,reals)])


# read_organism('E.Coli')
read_organism('Homo.Sapiens')
processed_codons,processed_polypeptides = preprocess_series(*get_codons_and_polypeptides())
relative_bias = calc_relative_index_bias(processed_codons)
codon_with_max_bias = max_bias_per_index(relative_bias)
# max_values = max_bias_codon_per_index(relative_bias,max_bias_per_index(relative_bias))
print(predict_organism(processed_codons,processed_polypeptides,codon_with_max_bias))

# organisms = ['E.Coli','Drosophila.Melanogaster','Homo.Sapiens']
# organism = pd.read_pickle('C:/Users/nilsr/Documents/HS-Mannheim/IM1/PMDS/Codons/data/E.Coli/cleanedData.pkl')
# codon_sequences = codon_sequences.apply(lambda x: [str(x[i:i+3]) for i in range(0,len(x)-3,3)])