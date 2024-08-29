import argparse
import sys

from pftools.probe_design.smfish_probe_design import SMFishProbeDesigner
from pftools.probe_design.util import *
import pftools.probe_design.probe_design_paths as pdp

def parse_args(args):
    parser = argparse.ArgumentParser(description='Design smFISH probes for a set of genes.')
    parser.add_argument('gene_list', type=str, help='Path to a file containing a list of gene symbols and bit names.')
    parser.add_argument('bit_seqs', type=str, help='Path to a file containing a list of bit names and sequences.')
    parser.add_argument('output_prefix', type=str, help='Name of the output prefix')

    parser.add_argument('--output_type', type=str, default='oPool', help='Type of the output file. Can be oPool or 96well.')
    parser.add_argument('--n_probes_per_transcript', type=int, default=48, help='Number of probes to design per transcript.') 
    parser.add_argument('--pb_len', type=int, default=30, help='Length of the probes.')
    parser.add_argument('--gc_range', type=float, nargs=2, default=[25, 75], help='Range of GC content.') 
    parser.add_argument('--tm_range', type=float, nargs=2, default=[66, 76], help='Range of melting temperature.')
    parser.add_argument('--na_conc_molar', type=float, default=0.3, help='Concentration of Na+ ions in molar.')
    parser.add_argument('--extra_seqs_path', type=str, default=None, help="Path to a FASTA file containing extra sequences to design probes for.")
    parser.add_argument('--offtarget_ncrna_upper_bound', type=float, default=0.5, help='Upper bound on the fraction of off-targets allowed to ncRNAs.')
    parser.add_argument('--offtarget_mrna_upper_bound', type=float, default=75, help='Upper bound on the fraction of off-targets allowed to the transcriptome.')
    parser.add_argument('--bit_revcomp', action='store_true', help='Whether to reverse complement the bits.')
    parser.add_argument('--add_three_prime_bit', action='store_true', help='Whether to add the bit to the 3\' end of the probe.')
    parser.add_argument('--spacer', type=str, default='', help='Spacer sequence to add between the bit and the target sequence.')
    parser.add_argument('--n_repeats', type=int, default=1, help='Number of times to repeat the bit sequence.')

    parser.add_argument('--species', type=str, default='human', help='Species to design probes for.')
    #parser.add_argument('--data_path', type=str, default=pdp.BASE_GENOMES_PATH, help='Base path to the transcriptome/OTTable data directory.')

    parsed_args = parser.parse_args(args)
    return parsed_args

def main(args):
    # Get species-specific paths for the transcriptome and OTTables
    if args.species == 'human':
        transcriptome_path = pdp.HUMAN_CDNA_FASTA
        transcriptome_ottable_path = pdp.HUMAN_OTTABLE_17
        ncrna_transcriptome_ottable_path = pdp.HUMAN_OTTABLE_NCRNA_15
    elif args.species == 'mouse':
        transcriptome_path = pdp.MOUSE_CDNA_FASTA
        transcriptome_ottable_path = pdp.MOUSE_OTTABLE_17
        ncrna_transcriptome_ottable_path = pdp.MOUSE_OTTABLE_NCRNA_15

    # these are hardcoded based on the transcriptome and OTTable versions
    ncrna_k = 15
    mrna_k = 17
    designer = SMFishProbeDesigner(args.gene_list, args.bit_seqs, transcriptome_path, transcriptome_ottable_path, ncrna_transcriptome_ottable_path, n_probes_per_transcript=args.n_probes_per_transcript, extra_seq_path=args.extra_seqs_path, mrna_k=mrna_k, ncrna_k=ncrna_k)
    designer.design_probes_for_genes(args.pb_len, args.gc_range, args.tm_range, args.na_conc_molar)
    designer.add_bits_to_probes(args.bit_revcomp, args.add_three_prime_bit, args.spacer, args.n_repeats)

    if args.output_type == 'oPool':
        designer.save_as_opool_excel(args.output_prefix, args.output_prefix + "_opool.xlsx")
    elif args.output_type == '96well':
        designer.save_as_96well_plate(args.output_prefix + "_96well.xlsx")
    # save the probe dictionary as a CSV file
    designer._probe_table.to_csv(args.output_prefix + '.csv', index=False)

if __name__ == "__main__":
    parsed_args = parse_args(sys.argv[1:])
    main(parsed_args)