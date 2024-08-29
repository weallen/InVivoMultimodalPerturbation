# Quick probe design based on Bogdan's code
# For designing against individual sequences
# or for screening short sequences
# NOT for designing full MERFISH libraries
from typing import List, Dict
import os
import tempfile
import pftools.probe_design.LibraryDesign.LibraryDesigner as ld
import pftools.probe_design.LibraryDesign.LibraryTools as lt

BASE_GENOMES_PATH = "/faststorage/genomes/zyto/data" # update this to wherever your genomes are stored
HUMAN_MRNA_FASTA = os.path.join(BASE_GENOMES_PATH, "Homo_sapiens.GRCh38.mrna.fa")
HUMAN_NCRNA_FASTA = os.path.join(BASE_GENOMES_PATH, "Homo_sapiens.GRCh38.ncrna.fa")
MOUSE_MRNA_FASTA = os.path.join(BASE_GENOMES_PATH, "mm10_rna.fasta")
MOUSE_NCRNA_FASTA = os.path.join(BASE_GENOMES_PATH, "rtRNA.fasta")

MOUSE_CTTABLE_17 = os.path.join(BASE_GENOMES_PATH, "mm10_rna_17w.npy")
MOUSE_CTTABLE_NCRNA_15 = os.path.join(BASE_GENOMES_PATH, "rtRNA_15w.npy")
MOUSE_CTTABLE_NCRNA_17 = os.path.join(BASE_GENOMES_PATH, "rtRNA_17w.npy")
HUMAN_CTTABLE_17 = os.path.join(BASE_GENOMES_PATH, "hg38_rna_17w.npy")
HUMAN_CTTABLE_NCRNA_15 = os.path.join(BASE_GENOMES_PATH, "Homo_sapiens.GRCh38.ncrna_15w.npy")
HUMAN_CTTABLE_NCRNA_17 = os.path.join(BASE_GENOMES_PATH, "Homo_sapiens.GRCh38.ncrna_17w.npy")

# Use Bogdan's code to design probes for specific sequences
# Usually used to design probes against barcodes or exogenous genes like GFP rather than for endogenous mRNA
def design_probes_for_seqs(seqs:Dict[str, str], mrna_cttable_file:str=MOUSE_CTTABLE_17, rep_cttable_file:str=MOUSE_NCRNA_FASTA,
                          n_offtarget:int=75, n_offtarget_rep:int=0, word_size:int=17, pb_len:int=30, gc_range:List[float]=[0.25, 0.75],
                           min_tm:int=70):
    """
    Design probes for a given sequence.
    """
    temp_file = "/tmp/temp.fa"
    with open(temp_file, 'w') as f:
        for k,v in seqs.items():
            f.write(f">{k}\n")
            f.write(f"{v}\n")

        pb_designer = ld.pb_reports_class(
            sequence_dic={'file':temp_file,'use_revc':False,'use_kmer':True},
            map_dic={'genome':{'file':mrna_cttable_file,'use_revc':False,'use_kmer': True},
                'rep_genome':{'file':rep_cttable_file,'use_revc':True,'use_kmer':True},
                'input':{'file':temp_file,'use_revc':False,'use_kmer':True}},
            params_dic={'word_size':word_size,'pb_len':pb_len,'buffer_len':2,'max_count':2**16-1,  'check_on_go':False,'auto':False},
            dic_check={('genome','input'):n_offtarget,#how many 17-mers offtarget       allowed 75-150 is reasonable range against the genome
                        'rep_genome':n_offtarget_rep,# how many 17-mer max hits to the repetitive     genome
                    'gc':gc_range,#gc range
                    'tm':min_tm})

        pb_designer.computeOTmaps()
        pb_designer.compute_pb_report()
        pb_designer.perform_check_end()
        return pb_designer

