from Bio.Data import CodonTable
import pandas as pd
import numpy as np

organisms = ["E.Coli"]#"Drosophila.Melanogaster", "Homo.Sapiens"

dfs = {}

def group_codons(sequence):
    return [''.join(sequence[i:i+3]) for i in range(0, len(sequence), 3)]


def getDict(codon_sequence, keys):
    synonymeCodonsDict = {}
    for k in keys:
        synonymeCodonsDict[k] = [c for c in codon_sequence if c in translation_table[k]]
    synonymeCodonsDict = {k:v for k,v in synonymeCodonsDict.items() if len(v)>1}
    return synonymeCodonsDict



def getChangeProbability(codonsDict,tabelle):
    lenTabelle = len(tabelle)
    tabelle.loc[lenTabelle]=np.NaN
    for k,v in codonsDict.items():
        unique, counts = np.unique(codonsDict[k][:-1], return_counts=True)
        counts = dict(zip(unique, counts))
        for i in range(1,len(v)):
            if(pd.isna(tabelle.loc[lenTabelle,codonsDict[k][i]])): 
                    tabelle.loc[lenTabelle,codonsDict[k][i]] = 0
            if (codonsDict[k][i] == codonsDict[k][i-1]):
                tabelle.loc[lenTabelle,codonsDict[k][i]] += 1/(counts[codonsDict[k][i]])
    
    #tabelle = tabelle.fillna(0)
    return tabelle

def getChangeProbabilityForOrganism(df):
    tabelle = pd.DataFrame(columns=[
        
    'AAA', 'AAG', 'AAT', 'AAC', 'AGA', 'AGG', 'AGT', 'AGC',
    'ATA', 'ATG', 'ATT', 'ATC', 'ACA', 'ACG', 'ACT', 'ACC',
    'GAA', 'GAG', 'GAT', 'GAC', 'GGA', 'GGG', 'GGT', 'GGC',
    'GTA', 'GTG', 'GTT', 'GTC', 'GCA', 'GCG', 'GCT', 'GCC',
    'TAA', 'TAG', 'TAT', 'TAC', 'TGA', 'TGG', 'TGT', 'TGC',
    'TTA', 'TTG', 'TTT', 'TTC', 'TCA', 'TCG', 'TCT', 'TCC',
    'CAA', 'CAG', 'CAT', 'CAC', 'CGA', 'CGG', 'CGT', 'CGC',
    'CTA', 'CTG', 'CTT', 'CTC', 'CCA', 'CCG', 'CCT', 'CCC'

])
 
    for i in range(3000,len(df)):
        d = getDict(df['codons'].iloc[i],df['translation'].iloc[i])
        tabelle = getChangeProbability(d,tabelle)
        
    return tabelle

genetic_code = CodonTable.unambiguous_dna_by_id[1]
translation_table = {}
for i in genetic_code.forward_table:
    translation_table.setdefault(genetic_code.forward_table[i], []).append(i)
 
translation_table['*'] = ['TAA', 'TAG', 'TGA']



for organism in organisms:
    dfs[organism] = pd.read_pickle(f"data/{organism}/cleanedData.pkl")
    
    dfs[organism]['codons'] = dfs[organism]['sequence'].apply(group_codons)
    result = getChangeProbabilityForOrganism(dfs[organism])
    
    #result.to_pickle(f"data/{organism}/changeTable.pkl")
    pickle = pd.read_pickle(f"data/{organism}/changeTable.pkl")
    result = pd.concat([pickle,result])
    result.to_pickle(f"data/{organism}/changeTable.pkl")

