import argparse
import pandas as pd
import numpy as np
from MERFISH_probe_design.IO.file_io import write_merlin_codebook
from pftools.probe_design.util import load_transcriptome
from pftools.probe_design.codebook import generate_initial_codebook_assignment

def main():
    parser = argparse.ArgumentParser(description='Generate a codebook assignment file for a set of genes.')
    parser.add_argument('gene_list', type=str, help='Path to a file containing a list of gene symbols.')
    parser.add_argument('codebook', type=str, help='Path to the codebook file, as a numpy matrix.')
    parser.add_argument('bit_path', type=str, help='Path to a file containing a list of bit names and sequences.') 
    parser.add_argument('--min_bit', type=int, default=0, help='Minimum bit index to use. NOTE: 0-indexed!!!')
    parser.add_argument('--output_prefix', type=str, default='codebook', help='Name of the output.')
    parser.add_argument('--transcriptome_path', type=str, default=None, help='Path to the transcriptome FASTA file.')

    args = parser.parse_args()

    # Read the gene list
    gene_list = pd.read_csv(args.gene_list)
    n_genes = gene_list.shape[0]

    # Read the codebook as a numpy array
    codes = np.load(args.codebook)
    n_codes, n_code_bits = codes.shape

    if n_genes > n_codes:
        print("Error: the number of genes in the gene list is greater than the number of genes in the codebook.")
        return

    # Read the bit names and sequences
    bit_names = pd.read_csv(args.bit_path)['bit_name'].tolist()
    # start the bit indexing at a different place
    if args.min_bit > 0:
        bit_names = bit_names[int(args.min_bit):]

    n_bits = len(bit_names)
    if n_code_bits > n_bits:
        print("Error: the number of bits in the codebook is greater than the number of bits in the bit file.")
        return

    # Generate the codebook assignment file
    print(f"Generating codebook for {n_genes} genes and {n_code_bits} bits. {n_codes} codes in the codebook.")
    #print(gene_list['gene_symbol'].tolist())
    codebook = generate_initial_codebook_assignment(gene_list['gene_symbol'].tolist(), codes)

    # Save the codebook assignment file as a MERlin codebook
    codebook_file = args.output_prefix + '.csv'
    codebook_version = '1.0'
    codebook_name = args.output_prefix
    # get the names of the bits that are actually used based on 
    used_bit_idx = np.argwhere(np.any(codes, axis=0)).ravel()
    bit_names_used = [bit_names[i] for i in used_bit_idx]

    # Get the transcript IDs if the transcriptome is provided
    if args.transcriptome_path is not None:
        transcriptome = pd.read_csv(args.transcriptome_path)
        transcript_ids = []
        for gene_symbol in codebook['name'].tolist():
            curr_tx = transcriptome[transcriptome.gene_short_name == gene_symbol]
            if curr_tx.shape[0] > 0:
                transcript_ids.append(curr_tx.transcript_id.iloc[0])
            else:
                transcript_ids.append('')
    else:
        transcript_ids = codebook['name'].tolist()
    # Save the codebook
    #print(codebook['name'].tolist())
    write_merlin_codebook(codebook_file, codebook_version, codebook_name, bit_names_used, 
                          codebook['name'].tolist(), transcript_ids, 
                          codebook.barcode_str.tolist())
    
if __name__ == "__main__":
    main()