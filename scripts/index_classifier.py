import pandas as pd
import numpy as np
import Bio.Seq as Bio

def read_organism(organism_name):
    global organism
    global global_cub
    global_cub = pd.read_pickle(f"C:/Users/nilsr/Documents/HS-Mannheim/IM1/PMDS/Codons/data/{organism_name}/usageBias.pkl")
    organism = pd.read_pickle(f"C:/Users/nilsr/Documents/HS-Mannheim/IM1/PMDS/Codons/data/{organism_name}/cleanedData.pkl")

def get_codons_and_polypeptides()->tuple[pd.Series,pd.Series]:
    return pd.Series(organism['sequence']),pd.Series(organism['translation'])

def mu_sigma_interval_length():
    mu = np.mean(organism['sequence'].apply(len)//3)//1
    sigma = np.std(organism['sequence'].apply(len)//3)//1
    return mu,sigma

def preprocess_series(codons:pd.Series,polypeptide:pd.Series,min=0)->tuple[pd.Series,pd.Series]:    
    mu,sigma = mu_sigma_interval_length()
    upper = mu+sigma
    #polypeptides_sample = polypeptide[polypeptide.apply(len).between(0,max(polypeptide.apply(len)),'both')]
    polypeptides_sample = polypeptide.apply(lambda x: x.seq)
    polypeptides_sample = polypeptides_sample.reset_index(drop=True)
    codon_sequences = codons.apply(lambda x: [str(x[i:i+3]) for i in range(0,len(x),3)])
    return codon_sequences,polypeptides_sample

"""
big_dict = {0:{'ATG':12},1:{'GCG':3,'GGG':1,...},...}
"""

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

def add_treshold_to_bias(relaitve_big_dict:dict):
    for index in relaitve_big_dict.keys():
        sub_dict = relaitve_big_dict[index]
        if len(sub_dict.keys()) < 10 and not index == 0:
            for aa in sub_dict.keys():
                sub_dict[aa] = (sub_dict[aa][1],global_cub[aa][sub_dict[aa][0]])
    return relaitve_big_dict

def calc_max_bias_per_aa(relative_big_dict):
    for index in range(len(relative_big_dict)):
        temp = relative_big_dict[index]
        temp2 = {}
        for k,v in temp.items():
            aa = Bio.translate(k)
            if not aa in temp2:
                temp2[aa] = (k,v)
            elif aa in temp2:
                if temp2[aa][1] <= v:
                    temp2[aa] = (k,v)
        relative_big_dict[index] = temp2

    return relative_big_dict

def calc_i_cub_per_chunk(processed_codons:pd.Series,chunk_width:int):
    dict_list = list()
    for length in range(0,len(processed_codons),chunk_width):
        if length + chunk_width > len(processed_codons):
            dict_list.append(calc_relative_index_bias(processed_codons[length:]))
        else:
            dict_list.append(calc_relative_index_bias(processed_codons[length:length+chunk_width]))
    return dict_list    

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
    for index in range(len(polypeptide)-1):
        predicted_sequence.append(max_on_index[index][polypeptide[index]][0])
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

organism_name = 'Homo.Sapiens'
read_organism(organism_name)
processed_codons,processed_polypeptides = preprocess_series(*get_codons_and_polypeptides())
relative_freq = calc_relative_index_bias(processed_codons)
stuff = calc_max_bias_per_aa(relative_freq)
print("Prediction Accuracy for ",organism_name,": ",predict_organism(processed_codons,processed_polypeptides,stuff))
thresholded = add_treshold_to_bias(relative_freq)
print("Prediction Accuracy for ",organism_name," after Threshold: ",predict_organism(processed_codons,processed_polypeptides,thresholded))
# relative_bias = calc_relative_index_bias(processed_codons)
# codon_with_max_bias = max_bias_per_index(relative_bias)
# print(predict_organism(processed_codons,processed_polypeptides,codon_with_max_bias))