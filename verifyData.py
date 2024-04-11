
from Bio import SeqIO
from Bio.SeqUtils import CheckSum


def verifyData(records) -> dict:
    removedRecords = []
    checkSum = []
    reportData = {
        'modThree': 0,
        'stopCodon':0,
        'doubleData': 0,
        'oldLength': len(records)
        }
    for val in records.values():
        seq = val.seq
        key = CheckSum.seguid(seq)
        #Überprüft ob Sequenz durch drei Teilbar
        if(len(seq)%3 != 0):
            removedRecords.append(val.id)
            reportData['modThree'] +=1
        #Überprüft ob Stop Codon nicht nur am Ende steht
        if('*' in seq.translate()[:-1]):
            removedRecords.append(val.id)
            reportData['stopCodon'] +=1
        #Prüft anhand des "SEquence Globally Unique IDentifier" ob Sequenz bereits da ist.
        if(key in checkSum):
            removedRecords.append(val.id)
            reportData['doubleData'] +=1
        else:
            checkSum.append(key)
    #Entfernt alle fehlerhaften Datensätze
    for key in removedRecords:
        del records[key]
    return {"records": records,"report":reportData}

records = SeqIO.to_dict(SeqIO.parse("data\E.Coli\GCA_000005845.2\cds_from_genomic.fna", "fasta"))

cleanRecords = verifyData(records=records)
print(cleanRecords['report'])
print(len(cleanRecords['records']))

