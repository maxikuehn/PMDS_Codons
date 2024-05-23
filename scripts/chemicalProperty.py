from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import product

# class BinaryEncodingChemical():
    ##  Binary Encoding for Chemical Properties of a Nucleiod Base Atom
    ##  2^2 Represents the Amount of Hydrogenbonds on the Base
    ##  100 = 3 Hydrogenbonds(G,C) --- 000 2 Hydogenbonds(A,T)
    ##  2^1 Represents if the Base is Purine or Pyrimidine
    ##  010 = Purine(A,G) --- 000 = Pyrimidine(T,C)
    ##  2^0 Represents if the Base is Keto or Amino
    ##  001 = Keto(G,T) --- 000 = Amino(A,C)
    ##  +-----+-----+-----+-----+
    ##  |  A  |  C  |  G  |  T  |
    ##  +-----+-----+-----+-----+
    ##  | 010 | 100 | 111 | 001 |
    ##  +-----+-----+-----+-----+
    ##  | 101 | 011 | 000 | 110 |  <-- Bitflip
    ##  +-----+-----+-----+-----+
#DONE -> Mithilfe dieses Encodings schauen ob die Codons mit dem höchsten CUB auch die mit dem höchsten Numerischen Wert sind (oder die am wenigsten vorkommenden)

def purineOrPyrimidine(codon:str)->str:
    return "".join([puriPyriDict[x] for x in codon])
    
def twoOrthreeHydrogenBonds(codon:str)->list:
    return "".join([str(hydrogenDict[x]) for x in codon])
    
def ketoOrAmino(codon:str)->str:
    return "".join([ketoAminoDict[x] for x in codon])

def encodeCodons(codon:str)->str:
    binaryEncodedBases = {'A':'010','C':'100','G':'111','T':'001'}
    return "".join([binaryEncodedBases[x] for x in codon])
     
def encodeCodonsInverse(codon:str)->str:
    inverseEncodedBases = {'A':'101','C':'011','G':'000','T':'110'}
    return "".join([inverseEncodedBases[x] for x in codon])

def scoreByChemProp(dataFrame,CODONLIST:list,inverse:bool):    
    df = dataFrame
    filteredByAA = []
    if not inverse:
        for index in range(len(df)):
            sequence = str(df.iloc[index].sequence)
            codon_score_pairs = [(sequence[x:x+3],int(encodeCodons(sequence[x:x+3]),2),twoOrthreeHydrogenBonds(sequence[x:x+3]),purineOrPyrimidine(sequence[x:x+3]),ketoOrAmino(sequence[x:x+3])) for x in range(0,len(sequence)-3,3)]
            sortedByEncoding = sorted(codon_score_pairs,key=lambda entry:entry[1])
            filteredByAA += [x for x in filter(lambda g: g[0] in CODONLIST,sortedByEncoding)]
    else:
        for index in range(len(df)):
            sequence = str(df.iloc[index].sequence)
            codon_score_pairs = [(sequence[x:x+3],int(encodeCodonsInverse(sequence[x:x+3]),2),twoOrthreeHydrogenBonds(sequence[x:x+3]),purineOrPyrimidine(sequence[x:x+3]),ketoOrAmino(sequence[x:x+3])) for x in range(0,len(sequence)-3,3)]
            sortedByEncoding = sorted(codon_score_pairs,key=lambda entry:entry[1])
            filteredByAA += [x for x in filter(lambda g: g[0] in CODONLIST,sortedByEncoding)]

    return pd.DataFrame(data=filteredByAA,columns=['CODON','SCORE','HYDROGEN','PURI_PYRI','KETO_AMINO'])

def scoreByChemPropContext(PATH:str,CODONLIST:list,vicinity:int):
    df = pd.read_pickle(PATH)
    filteredByAA = []
    vicinity*=3
    for index in range(len(df)):
        sequence = str(df.iloc[index].sequence)
        vicinity_score_pairs = [(sequence[x-vicinity:x+3+vicinity],int(encodeCodons(sequence[x-vicinity:x+3+vicinity]),2)) for x in range(vicinity,(len(sequence)-(3+vicinity)),3)]
        sortedByEncoding = sorted(vicinity_score_pairs,key=lambda entry:entry[1])
        unfilteredByAA = [[subseq for _ in range(0,len(subseq[0])-3,3) if subseq[0][vicinity:vicinity+3] in CODONLIST]for subseq in sortedByEncoding]
        unfilteredByAA = [entry for entry in unfilteredByAA if not entry == []]
        filteredByAA += [item for row in unfilteredByAA for item in row]
    average_value = sum([x[1] for x in filteredByAA])/len(filteredByAA)
    vicinityDataByAA = pd.DataFrame(data=filteredByAA,columns=['subsequence','score'])
    vicinityDataByAA['off_avg'] = (vicinityDataByAA['score']/average_value)
    return vicinityDataByAA
    
def setupCodonMatrix():
    labels = np.array(["".join(x)for x in product('ACGT',repeat=3)])
    codon_mapping_dict = {i+1:labels[i] for i in range(len(labels))}
    matrix = np.zeros(shape=(65,65),dtype=int)
    matrix[0,1:] = np.arange(1,65)
    matrix[1:,0] = np.arange(1,65)
    return matrix,codon_mapping_dict

def setupAminoMatrix():
    labels = ['A','C','D','E','F','G','H','I','K','L','M','N','P','Q','R','S','T','V','W','Y']
    amino_mapping_dict = {i+1:labels[i] for i in range(len(labels))}
    matrix = np.zeros(shape=(21,21),dtype=int)
    matrix[0,1:] = np.arange(1,21)
    matrix[1:,0] = np.arange(1,21)
    return matrix,amino_mapping_dict

def returnIndex(mapping_dict:dict,subseq:str):
    index = [x[0] for x in mapping_dict.items() if x[1]==subseq]
    return index[0]

def addRuns(matrix,seq:str,mapping_dict:dict,stepWidth:int):
    for x in range(0,len(seq)-2*stepWidth,stepWidth): #x ist ein Codon oder ein IUPAC AA Code
        i = returnIndex(mapping_dict,seq[x:x+stepWidth])
        j = returnIndex(mapping_dict,seq[x+stepWidth:x+(2*stepWidth)])
        matrix[i,j]+=1   
    return matrix

def calcTransitionRatio(matrix):
    all_transitions = np.sum(matrix[1:,1:])
    soloRatio = np.arange(len(matrix)-1)
    #berechne relative häufigkeit der einzelnen vorkommen
    for index in range(1,len(matrix)):
        soloRatio[index-1] = (np.sum(matrix[index,:])+np.sum(matrix[:,index]))-matrix[index,index]
    soloRatio = [element / all_transitions for element in soloRatio]
    #berechne relative häufigkeit "c1 impliziert c2" bzw. auf c1 folgt c2
    implicitRatio = np.zeros(shape=matrix.shape)
    implicitRatio[0,:] = np.arange(len(matrix))
    implicitRatio[:,0] = np.arange(len(matrix))
    for index in range(1,len(matrix)):
        for leftover in range(1,len(matrix)):
            rhc1_c2 = matrix[index,leftover]/all_transitions
            implicitRatio[index,leftover] = rhc1_c2 / soloRatio[index-1]*soloRatio[leftover-1]
    return implicitRatio


def kindaBLOSUMcodons(sequences:list):
    matrix,codon_mapping_dict = setupCodonMatrix()
    codonList = [str(x) for x in codon_mapping_dict.values()]
    for sequence in sequences:
        matrix = addRuns(matrix,sequence,codon_mapping_dict,3)
    transitionRatios = calcTransitionRatio(matrix)
    fig,ax = plt.subplots()
    fig.set_size_inches(12,12)
    sns.heatmap(transitionRatios[1:,1:],annot=False,cbar=True,cmap='mako',linewidths=.05,xticklabels=codonList,yticklabels=codonList)
    ax.tick_params(labelsize=5)
    ax.set_title("Nachbarschafts Abhängigkeit der Codons")
    plt.show()

def kindaBLOSUMaminos(sequences:list,plot_title:str):
    matrix,amino_mapping_dict = setupAminoMatrix()
    aminoList = [str(x) for x in amino_mapping_dict.values()]
    for sequence in sequences:
        matrix = addRuns(matrix,sequence,amino_mapping_dict,1)   
    transtitionRatios = calcTransitionRatio(matrix)
    fig,ax = plt.subplots()    
    sns.heatmap(transtitionRatios[1:,1:],annot=False,cbar=True,cmap='mako',linewidths=.5,xticklabels=aminoList,yticklabels=aminoList)
    ax.set_title(plot_title)
    plt.show()


PATH = "data/E.Coli/cleanedData.pkl"
puriPyriDict = {'C':'Y','G':'P','A':'P','T':'Y'}
hydrogenDict = {'G':3,'C':3,'A':2,'T':2}
ketoAminoDict = {'A':'A','C':'A','T':'K','G':'K'}
aminoDecoding = {'M': ['ATG'],
 'K': ['AAA', 'AAG'],
 'R': ['CGC', 'CGA', 'CGT', 'AGG', 'CGG', 'AGA'],
 'I': ['ATT', 'ATC', 'ATA'],
 'S': ['AGC', 'TCA', 'TCT', 'AGT', 'TCG', 'TCC'],
 'T': ['ACC', 'ACA', 'ACG', 'ACT'],
 'G': ['GGT', 'GGC', 'GGG', 'GGA'],
 'N': ['AAC', 'AAT'],
 'A': ['GCG', 'GCA', 'GCC', 'GCT'],
 '*': ['TGA', 'TAA', 'TAG'],
 'V': ['GTG', 'GTT', 'GTC', 'GTA'],
 'L': ['TTG', 'CTG', 'CTC', 'TTA', 'CTT', 'CTA'],
 'F': ['TTC', 'TTT'],
 'E': ['GAA', 'GAG'],
 'D': ['GAT', 'GAC'],
 'Q': ['CAG', 'CAA'],
 'P': ['CCC', 'CCG', 'CCT', 'CCA'],
 'H': ['CAC', 'CAT'],
 'C': ['TGC', 'TGT'],
 'Y': ['TAC', 'TAT'],
 'W': ['TGG']}

# df = pd.read_pickle(PATH)
# codon_sequences = df['sequence'].tolist()
# kindaBLOSUMcodons(codon_sequences)
# amino_sequences = df['translation'].apply(lambda x: x.seq).tolist()
# kindaBLOSUMaminos(amino_sequences,'E.Coli')
# scoreByChemProp(df,aminoDecoding['I'],True)
