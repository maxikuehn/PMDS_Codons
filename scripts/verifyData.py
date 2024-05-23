
from Bio import SeqIO
from Bio.SeqUtils import CheckSum


def verifyData(records) -> dict:
    removedRecords = []
    checkSum = []
    reportData = {
        'modThree': 0,
        'stopCodon':0,
        'doubleData': 0,
        'noStartCodon':0,
        'oldLength': len(records)
        }
    for val in records.values():
        seq = val.seq
        key = CheckSum.seguid(seq)
        #Überprüft ob Sequenz durch drei Teilbar
        if not len(seq)%3 == 0:
            removedRecords.append(val.id)
            reportData['modThree'] +=1
        #Überprüft ob Stop Codon nicht nur am Ende steht
        elif('*' in seq.translate()[:-1]):
            removedRecords.append(val.id)
            reportData['stopCodon'] +=1
        #Prüft anhand des "SEquence Globally Unique IDentifier" ob Sequenz bereits da ist.
        elif(key in checkSum):
            removedRecords.append(val.id)
            reportData['doubleData'] +=1
        #Prüft ob Sequenz *nicht* mit StartCodon 'ATG' beginnt.
        elif not seq[0:3] == "ATG":
            removedRecords.append(val.id)
            reportData['noStartCodon']+=1
        else:
            checkSum.append(key)
    #Entfernt alle fehlerhaften Datensätze
    [record for record in records if record not in removedRecords]
    return {"records": records,"report":reportData}

recordsEColi = SeqIO.to_dict(SeqIO.parse("Codons\data\E.Coli\GCA_000005845.2\cds_from_genomic.fna", "fasta"))
# recordsFly = SeqIO.to_dict(SeqIO.parse("data\Drosophila.Melanogaster\cds_from_genomic.fna", "fasta"))
# recordsHuman= SeqIO.to_dict(SeqIO.parse("data\Homo.Sapiens\cds_from_genomic.fna", "fasta"))

# Remove unwanted Symbols from Aminoacidsequences('X') or Codonsequences ('*' at last index)
def removeUnwantedSymbols(sequences):
    newList = []
    for sequence in sequences:
        if "X" in sequence:
            continue
        elif sequence[-1] == '*':
            newList.append(sequence[:-2])
        else:
            newList.append(sequence)
    return newList

def printCleanRecord(record):
    cleanRecords = verifyData(records=record)
    print(cleanRecords['report'])
    print(len(cleanRecords['records']))   

printCleanRecord(recordsEColi)
# printCleanRecord(recordsFly)
# printCleanRecord(recordsHuman)