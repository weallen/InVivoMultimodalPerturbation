import os
BASE_GENOMES_PATH = "/faststorage/genomes/perturbfish/" # update this to wherever your genomes are stored
SEQUENCES_PATH = os.path.join(BASE_GENOMES_PATH, "sequences")
HUMAN_CDNA_FASTA = os.path.join(SEQUENCES_PATH, "Homo_sapiens.GRCh38.cdna.longest_transcripts.csv")
HUMAN_NCRNA_FASTA = os.path.join(SEQUENCES_PATH, "Homo_sapiens.GRCh38.ncrna.fa")
MOUSE_CDNA_FASTA = os.path.join(SEQUENCES_PATH, "Mus_musculus.GRCm39.cdna.longest_transcripts.csv")
MOUSE_NCRNA_FASTA = os.path.join(SEQUENCES_PATH, "Homo_sapiens.GRCh38.ncrna.fa")
OTTABLES_PATH = os.path.join(BASE_GENOMES_PATH, "ottables")
MOUSE_OTTABLE_17 = os.path.join(OTTABLES_PATH, "Mus_musculus.GRCm39.cdna_17w.pkl")
MOUSE_OTTABLE_NCRNA_15 = os.path.join(OTTABLES_PATH, "Mus_musculus.GRCm39.ncrna_15w.pkl")
HUMAN_OTTABLE_17 = os.path.join(OTTABLES_PATH, "Homo_sapiens.GRCh38.cdna_17w.pkl")
HUMAN_OTTABLE_NCRNA_15 = os.path.join(OTTABLES_PATH, "Homo_sapiens.GRCh38.ncrna_15w.pkl")
