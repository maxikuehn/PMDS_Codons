from database import add_fasta_to_db, ping_db

ping_db()
add_fasta_to_db("data/GCA_900166955.1.fasta", organism="Escherichia coli")
