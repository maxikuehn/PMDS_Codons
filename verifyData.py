
from Bio import SeqIO
from Bio.SeqUtils import CheckSum
import itertools


def verifyData(records):
    removedRecords = []
    modThree = []
    doubleData = []
    stopCodon = []
    checkSum = []
    oldLength = len(records)
    for val in records.values():
        seq = val.seq
        key = CheckSum.seguid(seq)
        if(len(seq)%3 != 0):
            removedRecords.append(val.id)
            modThree.append(val.id)
        if(key in checkSum):
            removedRecords.append(val.id)
            doubleData.append(val.id)
        if('*' in seq.translate()[:-1]):
            removedRecords.append(val.id)
            stopCodon.append(val.id)
        else:
            checkSum.append(key)

    for key in itertools.chain(modThree,doubleData,stopCodon):
        del records[key]
    return {"records": records,"modThree":modThree,"doubleData": doubleData,"stopCodon": stopCodon, "oldLength": oldLength}

records = SeqIO.to_dict(SeqIO.parse("data\E.Coli\GCA_000005845.2\cds_from_genomic.fna", "fasta"))

cleanRecords = verifyData(records=records)
print(len(cleanRecords["records"])/cleanRecords["oldLength"])
print(len(cleanRecords["modThree"])/cleanRecords["oldLength"])
print(len(cleanRecords["doubleData"])/cleanRecords["oldLength"])
print(len(cleanRecords["stopCodon"])/cleanRecords["oldLength"])
