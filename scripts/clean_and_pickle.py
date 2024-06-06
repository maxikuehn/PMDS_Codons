from Bio import SeqIO
from Bio.SeqUtils import CheckSum
import pandas as pd

raw = pd.DataFrame(columns=["id","description","sequence","translation","seguid"])
reportData = {
        'modThree': 0,
        'stopCodon':0,
        'doubleData': 0,
        'noStartCodon':0,
        #'oldLength': len(records)
        }
for index,record in enumerate(SeqIO.parse("C:/Users/nilsr/Documents/HS-Mannheim/IM1/PMDS/Codons/data/Homo.Sapiens/cds_from_genomic.fna", "fasta")):
    if not len(record.seq)% 3 == 0:
        reportData['modThree']+=1
        continue
    elif not record.seq[0:3] == 'ATG':
        reportData['noStartCodon']+=1
        continue
    elif "*" in record.translate()[:-1]:
        reportData['stopCodon']+=1
        continue
    else:
        raw.loc[index] = {'id':record.id,'description':record.description,'sequence':record.seq,'translation':record.translate(),'seguid':CheckSum.seguid(record.seq)}

for index,row in raw.iterrows():
    if "X" in row['translation']:
        raw.drop(index=index,inplace=True)
    elif "*" == row['translation'][-1]:
        raw.loc[index,'translation'] = raw.loc[index,'translation'][:-1]
        raw.loc[index,'sequence'] = raw.loc[index,'sequence'][:-3]
        assert len(raw.loc[index,'sequence']) / 3 == len(raw.loc[index,'translation'])        
        
before_dupe_drop = raw.shape[0]
raw.drop_duplicates(subset=['seguid'],inplace=True)
after_dupe_drop = raw.shape[0]
reportData['doubleData'] = before_dupe_drop - after_dupe_drop
reportDataDf = pd.DataFrame(reportData,index=[1,2,3,4])
raw.to_pickle('C:/Users/nilsr/Documents/HS-Mannheim/IM1/PMDS/Codons/data/Homo.Sapiens/cleanedData.pkl')
reportDataDf.to_pickle('C:/Users/nilsr/Documents/HS-Mannheim/IM1/PMDS/Codons/data/Homo.Sapiens/reportData.pkl')