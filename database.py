import os

from Bio.SeqRecord import SeqRecord
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from Bio import SeqIO

load_dotenv()
MONGO_DB_USER = os.getenv("MONGO_DB_USER")
MONGO_DB_PASSWORD = os.getenv("MONGO_DB_PASSWORD")

uri = f"mongodb+srv://{MONGO_DB_USER}:{MONGO_DB_PASSWORD}@codondb.xtmmewu.mongodb.net/?retryWrites=true&w=majority&appName=CodonDB"

# Create a new client and connect to the server
client = MongoClient(uri)


def ping_db():
    """Send a ping to confirm a successful connection"""
    try:
        client.admin.command('ping')
        print("Pinged your deployment. You successfully connected to MongoDB!")
    except Exception as e:
        print(e)


def create_record(record: SeqRecord, species, organism):
    seq = record.seq

    gap_count = len(seq) % 3
    # seq += gap_count * "N"
    if gap_count > 0:
        seq = seq[:-gap_count]

    try:
        translation = seq.translate(gap='-')
    except Exception as e:
        print("Translation Error:", e)
        return None

    # Codon Match sprengt leider gelegentlich das MongoDB Upload Limit
    # codon_match = []
    # for i, t in enumerate(translation):
    #     codon_match.append({str(seq[i * 3:i * 3 + 3]): str(t)})

    return {
        "id": record.id,
        "description": record.description,
        "species": species,
        "organism": organism,
        "codon_sequence": str(seq),
        "translation": str(translation),
        # "codon_match": codon_match
        "processed_for_usage_bias": False
    }


def add_fasta_to_db(fasta_file: str, species="", organism=""):
    db = client.codondb
    collection = db.sequences

    records = []

    for record in SeqIO.parse(fasta_file, "fasta"):
        rec = create_record(record, species, organism)
        records.append(rec)

    try:
        collection.insert_many(records)
    except Exception as e:
        print(e)
        return

    print("Added {} records to DB".format(len(records)))
