import sys
from pftools.probe_design.merfish_probe_design import *
from pftools.probe_design.util import *
import pftools.probe_design.probe_design_paths as pdp

import argparse

def parse_args(args):
    # Parse arguments
    parser = argparse.ArgumentParser(description='Design SplintR probes for a set of genes.')
    parser.add_argument('codebook', type=str, help='Path to the codebook file.')
    parser.add_argument('bit_seqs', type=str, help='Path to a file containing a list of bit names and sequences.')
    parser.add_argument('output_prefix', type=str, help='Name of the output prefix')
    parser.add_argument('--pb_len', type=int, default=30, help='Length of the left probe.')
    parser.add_argument('--gc_range', type=float, nargs=2, default=[25, 75], help='Range of GC content.') 
    parser.add_argument('--tm_range', type=float, nargs=2, default=[66, 76], help='Range of melting temperature.')
    parser.add_argument('--na_conc_molar', type=float, default=0.3, help='Concentration of Na+ ions in molar.')
    parser.add_argument('--offtarget_ncrna_upper_bound', type=float, default=0.5, help='Upper bound on the fraction of off-targets allowed to ncRNAs.')
    parser.add_argument('--offtarget_mrna_upper_bound', type=float, default=50, help='Upper bound on the fraction of off-targets allowed to the transcriptome.')
    parser.add_argument('--bit_revcomp', type=bool, default=False, help='Whether to reverse complement the bits.')
    parser.add_argument('--primer_pair_path', type=str, default=None, help='Path to a file containing a list of primer pairs.')
    parser.add_argument('--primer_pair', type=str, default=None, help='Name of primer pair to use')
    parser.add_argument('--forward_primer', type=str, default='CGGCTCGCAGCGTGTAAACG', help='Sequence of the forward primer.')
    parser.add_argument('--reverse_primer', type=str, default='CATTTCAGGATCACCGGCGG', help='Sequence of the reverse primer.')
    parser.add_argument('--n_probes_per_transcript', type=int, default=96, help='Number of probes to design per transcript.')
    parser.add_argument('--extra_seqs', type=str, default=None, help='Path to a FASTA file containing extra sequences to design probes for.')
    parser.add_argument('--species', type=str, default='human', help='Species to design probes for.')
    parser.add_argument('--is_barcode', type=bool, default=False, help='Whether to design barcode probes, included as extra_seqs')
    parser.add_argument('--n_readouts_per_probe', type=int, default=4, help='Number of on readouts per probe')
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

    ncrna_k = 15
    mrna_k = 17
    designer = MERFISHProbeDesign(args.codebook, args.bit_seqs, transcriptome_path, transcriptome_ottable_path, ncrna_transcriptome_ottable_path, 
                                        n_probes_per_transcript=args.n_probes_per_transcript, extra_seq_path=args.extra_seqs, mrna_k=mrna_k, ncrna_k=ncrna_k)
    print("Designing probes...")
    if args.is_barcode:
        overlap = 10
    else:
        overlap = 1
    designer.design_probes_for_genes(args.pb_len, args.gc_range, args.tm_range, args.na_conc_molar, overlap=overlap)
    print("Adding bits to probes...")
    designer.add_bits_to_probes(n_readout_per_probe=args.n_readouts_per_probe, bit_revcomp=args.bit_revcomp)
    print("Adding primers to probes...")
    if args.primer_pair_path is None:
        forward_primer = args.forward_primer
        reverse_primer = args.reverse_primer
    else:
        primer_pairs = pd.read_csv(args.primer_pair_path)
        primer_pair = primer_pairs[primer_pairs['name'] == args.primer_pair]
        forward_primer = str(primer_pair['forward_primer'].values[0])
        reverse_primer = str(primer_pair['reverse_primer'].values[0])

    if args.is_barcode:
        designer.add_primers_to_probes(forward_primer, reverse_primer, subsample_probes=False)
    else:
        designer.add_primers_to_probes(forward_primer, reverse_primer, subsample_probes=True)
    print("Saving probes...")
    designer.save_csv_for_twist(args.output_prefix)

if __name__ == "__main__":
    parsed_args = parse_args(sys.argv[1:])
    main(parsed_args)