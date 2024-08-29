from Bio.Seq import Seq
from typing import List
from Bio import SeqIO
from Bio.SeqUtils import MeltingTemp
from tqdm import tqdm
import pandas as pd
import numpy as np
from multiprocessing import Pool

import MERFISH_probe_design.IO.file_io as fio
import MERFISH_probe_design.probe_design.OTTable_dict as ot
import MERFISH_probe_design.probe_design.quality_check as qc
import MERFISH_probe_design.probe_design.filters as filters

from pftools.probe_design.codebook import stringify_binary_mat_codebook

def calc_tm(seq:str, fmd:int=30, Na:int=300, Mg:int=0, type:str="dna/dna", probe_conc:float=1.0) -> float:
    """
    Calculate the corrected melting temperature of a sequence. 
    """
    if type == "dna/dna":
        nn_table = MeltingTemp.DNA_NN4
    elif type == "rna/dna":
        nn_table = MeltingTemp.R_DNA_NN1
    return MeltingTemp.chem_correction(MeltingTemp.Tm_NN(seq, Na=Na, Mg=Mg, nn_table=nn_table, dnac1=probe_conc, dnac2=0), fmd=fmd)

def gc(seq:str) -> float:
    """
    Calculate the GC content of a sequence. 
    """
    seq = seq.upper()
    return (seq.count('G') + seq.count('C')) / len(seq)

def seqrc(s):
    """
    Reverse complement a sequence. 
    """
    return str(Seq(s).reverse_complement())

def score_sequence_by_ottable(seq:str, ottable:ot.OTTable, K:int, score_type='sum') -> float:
    if score_type == 'mean':
        return np.mean([ottable[seq[i:i+K]] for i in range(len(seq) - K + 1)])
    elif score_type == 'sum':
        return np.sum([ottable[seq[i:i+K]] for i in range(len(seq) - K + 1)])
    return -1

def filter_probe_dict_by_fn(probe_dict:dict, fn, column_key_read:str, column_key_write:str): 
    for gk in probe_dict.keys():
        for tk in probe_dict[gk].keys():
            probe_dict[gk][tk][column_key_write] = probe_dict[gk][tk][column_key_read].apply(fn)
            new_df = probe_dict[gk][tk][~probe_dict[gk][tk][column_key_write]]
            print(f'\t{tk}: {new_df.shape[0]} / {probe_dict[gk][tk].shape[0]} probes passed the filter')
            probe_dict[gk][tk] = new_df

def filter_probe_dict(probe_dict:dict, column_key_seq:str, 
                      ncrna_ottable:ot.OTTable, transcriptome_ottable:ot.OTTable,
                      gc_range:List[float], tm_range:List[float], 
                      ncrna_k:int=15,
                      offtarget_ncrna_upper_bound:float=0.5,
                      mrna_k:int=17,
                      off_target_mrna_upper_bound:float=0.5,
                      na_conc_molar:int=0.3,
                      write_prefix:str="target"):
    filters.calc_gc_for_probe_dict(probe_dict, column_key_seq=column_key_seq, column_key_write=f'{write_prefix}_GC')
    filters.filter_probe_dict_by_metric(probe_dict, f'{write_prefix}_GC', lower_bound=gc_range[0], upper_bound=gc_range[1])

    # Filter by TM
    filters.calc_tm_JM_for_probe_dict(probe_dict, monovalentSalt=na_conc_molar, probe_conc=5e-9, column_key_seq=column_key_seq, column_key_write=f'{write_prefix}_Tm')
    filters.filter_probe_dict_by_metric(probe_dict, f'{write_prefix}_Tm', lower_bound=tm_range[0], upper_bound=tm_range[1])

    # Calculate and filter by off-targets to ncRNAs
    ot.calc_OTs(probe_dict, ncrna_ottable, column_key_seq, f'{write_prefix}_OT_ncRNA', int(ncrna_k))
    filters.filter_probe_dict_by_metric(probe_dict, f'{write_prefix}_OT_ncRNA', upper_bound=offtarget_ncrna_upper_bound)

    # Calculate and filter by off-targets to the transcriptome
    ot.calc_OTs(probe_dict, transcriptome_ottable, column_key_seq, f'{write_prefix}_OT_transcriptome', int(mrna_k))
    filters.filter_probe_dict_by_metric(probe_dict, f'{write_prefix}_OT_transcriptome', upper_bound=off_target_mrna_upper_bound)

# Step 1: Load the transcriptome. 
#
def load_ensembl_transcriptome(fpath, only_protein_or_lincrna=True):
    """
    Loads the transcriptome from an Ensembl FASTA file in the format needed for probe design. 
    """
    transcriptome = parse_fasta_cdna(fpath)
    # filter the transcriptome to preferentially include protein coding genes and lincRNAs
    if only_protein_or_lincrna:
        transcriptome = transcriptome[transcriptome.transcript_biotype.isin(["protein_coding", "lincRNA"])]
    transcriptome = transcriptome.loc[:, ["transcript_id", "gene","sequence", "gene_symbol"]]
    transcriptome.columns = ["transcript_id", "gene_id", "sequence", "gene_short_name"]
    transcriptome["FPKM"] = 1
    transcriptome = qc.check_and_standardize_transcriptome(transcriptome, remove_non_standard_columns=True)
    return transcriptome


def add_genes_to_transcriptome(transcriptome, gene_fasta_file):
    """
    Add additional genes to transcriptome. E.g. GFP or mCherry.
    """
    gene_fasta = fio.load_fasta_into_df(gene_fasta_file)
    gene_fasta.columns = ["gene_id", "transcript_id", "sequence"]
    gene_fasta["gene_short_name"] = gene_fasta["gene_id"]
    gene_fasta["FPKM"] = 1
    gene_fasta = qc.check_and_standardize_transcriptome(gene_fasta, remove_non_standard_columns=True)
    transcriptome = transcriptome.append(gene_fasta)
    return transcriptome

def load_transcriptome(fpath):
    transcriptome = fio.load_transcriptome(fpath)
    transcriptome['gene_short_name'] = [i.split(' ')[1][5:] for i in transcriptome['description']]
    transcriptome = qc.check_and_standardize_transcriptome(transcriptome, remove_non_standard_columns=True)
    return transcriptome

def find_median_length_transcript_for_gene(transcriptome):
    gene_names = transcriptome.gene_short_name.unique()
    symbol_to_transcript_id = {}
    for g in tqdm(gene_names):
        curr_isos = transcriptome[transcriptome.gene_short_name==g]
        # get the length of each isoform
        iso_lens = [len(i) for i in curr_isos.sequence]
        # identify the median length isoform
        median_iso_len = np.median(iso_lens)
        # find the transcript id for the median length isoform
        for i,r in curr_isos.iterrows():
            if len(r.sequence) == median_iso_len:
                symbol_to_transcript_id[g] = r.transcript_id
    return pd.DataFrame({'symbol':list(symbol_to_transcript_id.keys()),
                         'transcript_id':list(symbol_to_transcript_id.values())})


def find_longest_transcript_for_gene(transcriptome):
    gene_names = transcriptome.gene_short_name.unique()
    symbol_to_transcript_id = {}
    for g in tqdm(gene_names):
        curr_isos = transcriptome[transcriptome.gene_short_name==g]
        longest_iso_len = -np.inf
        for i,r in curr_isos.iterrows():
            if len(r.sequence) >= longest_iso_len:
                longest_iso_len = len(r.sequence)
                symbol_to_transcript_id[g] = r.transcript_id
    return pd.DataFrame({'symbol':list(symbol_to_transcript_id.keys()),
                         'transcript_id':list(symbol_to_transcript_id.values())})

def find_shortest_transcript_for_gene(transcriptome):
    gene_names = transcriptome.gene_short_name.unique()
    symbol_to_transcript_id = {}
    for g in tqdm(gene_names):
        curr_isos = transcriptome[transcriptome.gene_short_name==g]
        longest_iso_len = np.inf
        for i,r in curr_isos.iterrows():
            if len(r.sequence) < longest_iso_len:
                longest_iso_len = len(r.sequence)
                symbol_to_transcript_id[g] = r.transcript_id
    return pd.DataFrame({'symbol':list(symbol_to_transcript_id.keys()),
                         'transcript_id':list(symbol_to_transcript_id.values())})

def assign_shortest_transcript(gene_df, shortest_transcripts):
    """
    Assign shortest transcripts for list of genes and codewords 
    """
    ids = []
    for i in gene_df.name:
        if i in list(shortest_transcripts.symbol):
            ids.append(str(shortest_transcripts[shortest_transcripts.symbol==i].transcript_id.iloc[0]))
        else:
            ids.append("")
    gene_df['id'] = ids
    return gene_df

# Step 2: Compute the covering tables for the transcriptome and rRNA/tRNAs
#
def generate_initial_codebook_assignment(gene_names:List[str], codewords:np.ndarray, random_seed:int=40) -> pd.DataFrame:
    """
    Generate an initial codebook assignment with blanks, given a list of gene names.
    """

    blanks = [f'Blank-{n}' for n in range(codewords.shape[0]-len(gene_names))]
    gene_names_with_blanks = gene_names + blanks
    # set random seed
    np.random.seed(random_seed)
    codewords = codewords[np.random.permutation(codewords.shape[0]),:]
    
    return pd.DataFrame({'name': gene_names_with_blanks, 'barcode_str' : stringify_binary_mat_codebook(codewords[:len(gene_names_with_blanks),:])})

def get_gene_and_transcript_ids(barcode_table):
    gene_ids = list(barcode_table['name'][barcode_table['id'] != '']) # Get the non-blank gene names
    transcript_ids = set(barcode_table['id'][barcode_table['id'] != '']) # Get the non-blank transcript ids
    return gene_ids, transcript_ids

def compute_ncrna_ottable(ncrna_fasta, ottable_ncrna_file, word_len=15):
    ncRNAs =  fio.load_fasta_into_df(ncrna_fasta)
    ottable_rtRNAs = ot.get_OTTable_for_rtRNAs(ncRNAs, word_len)
    ottable_rtRNAs.save_pkl(ottable_ncrna_file)

def compute_transcriptome_ottable(transcriptome, ottable_transcriptome_file, word_len=17):
    ottable_transcriptome = ot.get_OTTable_for_transcriptome(transcriptome, word_len)
    ottable_transcriptome.save_pkl(ottable_transcriptome_file)    

# select probes trying to make as even as possible
def select_probes_nobits_greedy_stochastic_one_df(df:pd.core.frame.DataFrame, N_probes_per_transcript:int):
    '''A greedy stochastic method to select probes from one data frame.'''
    if N_probes_per_transcript >= df.shape[0]:
        print(f'There are only {df.shape[0]} probes while {N_probes_per_transcript} are required! Just return everything!')
        return df

    # Create a array to track the coverage of the transcript
    target_length = len(df.iloc[0]['target_sequence'])
    max_targetable_length = np.max(df['shift']) + target_length
    transcript_coverage = np.zeros(max_targetable_length)

    # Select probes
    selected_indices = []
    rest_indices = [i for i in range(df.shape[0])]
    for i in range(N_probes_per_transcript):
        
        # Calculate the scores if a probe is to be added
        trial_scores = []
        for r_id in rest_indices:
            # Calculate the number of overlaps
            shift = df.iloc[r_id]['shift']
            N_new_overlaps = np.sum([transcript_coverage[pos] for pos in range(shift, shift + target_length)])
            score = N_new_overlaps 
            trial_scores.append(score)

        score_min = min(trial_scores)
        lowest_score_ids = [rest_indices[j] for j in range(len(rest_indices)) if trial_scores[j] == score_min]

        # Randomly select an ID
        selected_id = np.random.choice(lowest_score_ids)
        selected_indices.append(selected_id)
        rest_indices.remove(selected_id)

        # Update the tracking records
        # Calculate the number of overlaps
        shift = df.iloc[selected_id]['shift']
        transcript_coverage[shift:shift + target_length] += 1
        

    #print(f'{df.iloc[0]["gene_id"]}:{df.iloc[0]["transcript_id"]}: selected {N_probes_per_transcript}/{df.shape[0]} probes with N_overlapping_bases={np.sum(transcript_coverage * (transcript_coverage - 1)  / 2)} and on-bit_coverage={on_bit_coverage}.')

    # Return a data frame with the selected indices
    return df.iloc[selected_indices]


def select_probes_greedy_stochastic_nobits(probe_dict:dict, N_probes_per_transcript:int, N_threads:int=1):
    '''A greedy stochastic method to select probes.
    Arguments:
        probe_dict: The dictionary of probes.
    '''
    keys = []
    args = []
    for gk in probe_dict.keys(): 
        for tk in probe_dict[gk].keys():
            keys.append((gk, tk))
            args.append([probe_dict[gk][tk], N_probes_per_transcript]) 
    
    with Pool(N_threads) as p:
        results = p.starmap(select_probes_nobits_greedy_stochastic_one_df, args)

    for i in range(len(keys)):
        gk, tk = keys[i]
        probe_dict[gk][tk] = results[i]

from Bio import SeqIO


def parse_fasta_cdna(fasta_file):
    """
    Parse a FASTA file with cDNA sequences and extract information from the headers.

    Args:
        fasta_file (str): Path to the FASTA file.

    Returns:
        list: A list of dictionaries containing the extracted information.
    """
    
    def parse_header(header):
        """
        Parse a FASTA header to extract information.

        Args:
            header (str): FASTA header string.
        
        Returns:
            dict: A dictionary with the extracted information.
        """
        fields = header.split(' ')
        info = {}
        info['transcript_id'] = fields[0].split(':')[0]
        info['sequence_type'] = fields[1]
        info['chromosome_info'] = fields[2]

        # Loop through the remaining fields to extract key-value pairs
        for field in fields[3:]:
            if ':' in field:
                key, value = field.split(':', 1)
                info[key] = value
            else:
                # Handle cases where there is no colon in the field
                info['additional_info'] = info.get('additional_info', '') + ' ' + field

        return info

    # Read the FASTA file and parse the headers
    records = []
    for record in SeqIO.parse(fasta_file, "fasta"):
        info = parse_header(record.description)
        info['sequence'] = str(record.seq)
        records.append(info)
    
    return pd.DataFrame(records).loc[:, ["transcript_id", "sequence_type", "chromosome_info", "gene", "gene_biotype", "transcript_biotype", "gene_symbol", "description", "additional_info", "sequence"]]
