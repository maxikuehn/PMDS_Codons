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
for index,record in enumerate(SeqIO.parse("Codons\data\Drosophila.Melanogaster\cds_from_genomic.fna", "fasta")):
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

before_dupe_drop = raw.shape[0]
raw.drop_duplicates(subset=['seguid'],inplace=True)
after_dupe_drop = raw.shape[0]
reportData['doubleData'] = before_dupe_drop - after_dupe_drop
reportDataDf = pd.DataFrame(reportData,index=[1,2,3,4])
raw.to_pickle('Codons/data/Drosophila.Melanogaster/cleanedData.pkl')
reportDataDf.to_pickle('Codons/data/Drosophila.Melanogaster/reportData.pkl')

# def cleanData(raw_df):
#     for index,record in raw_df.iterrows():
#         pass

#print(type(my_dict[next(iter(my_dict))].__dict__.keys()))
# for k,v in my_dict.items():
    # print("Key: %s "  "Value %s" %(k, v.__dict__.keys()))
#['_seq', 'id', 'name', 'description', 'dbxrefs', 'annotations', '_per_letter_annotations', 'features']