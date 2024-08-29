import sys
from pftools.probe_design.rca_probe_design import *
from pftools.probe_design.util import *
import pftools.probe_design.probe_design_paths as pdp

import argparse

def parse_args(args):
    # Parse arguments
    parser = argparse.ArgumentParser(description='Design SplintR probes for a set of genes.')
    parser.add_argument('gene_names', type=str, help='Path to the gene names file.') 
    parser.add_argument('output_prefix', type=str, help='Name of the output prefix')
    parser.add_argument('--pb_len_left', type=int, default=25, help='Length of the left probe.')
    parser.add_argument('--pb_len_right', type=int, default=25, help='Length of the right probe.')
    parser.add_argument('--gc_range', type=float, nargs=2, default=[44, 72], help='Range of GC content.') 
    parser.add_argument('--tm_range', type=float, nargs=2, default=[61, 71], help='Range of melting temperature.')
    parser.add_argument('--na_conc_molar', type=float, default=0.3, help='Concentration of Na+ ions in molar.')
    parser.add_argument('--offtarget_ncrna_upper_bound', type=float, default=0.5, help='Upper bound on the fraction of off-targets allowed to ncRNAs.')
    parser.add_argument('--offtarget_mrna_upper_bound', type=float, default=50, help='Upper bound on the fraction of off-targets allowed to the transcriptome.')
    parser.add_argument('--n_probes_per_transcript', type=int, default=3, help='Number of probes to design per transcript.')
    parser.add_argument('--extra_seqs', type=str, default=None, help='Path to a FASTA file containing extra sequences to design probes for.')
    parser.add_argument('--hyb_fmd', type=int, default=30, help="Concentration of formamide in hyb buffer")
    parser.add_argument('--species', type=str, default='human', help='Species to design probes for.')
    parser.add_argument('--is_barcode', action=argparse.BooleanOptionalAction, help='Whether to design barcode probes, included as extra_seqs')
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
    designer = FlexProbeDesigner(args.gene_names, transcriptome_path, transcriptome_ottable_path, ncrna_transcriptome_ottable_path, 
                                        n_probes_per_transcript=args.n_probes_per_transcript, extra_seq_path=args.extra_seqs, mrna_k=mrna_k, ncrna_k=ncrna_k)
    print("Designing probes...")
    overlap = 1
    designer.design_probes_for_genes(args.pb_len_left, args.pb_len_right, args.gc_range, args.tm_range, args.na_conc_molar, hyb_fmd=args.hyb_fmd, overlap=overlap, use_dnazyme=False)

    print("Adding primers to probes...")

    subsample = not args.is_barcode
    designer.add_primers_to_probes(subsample_probes=subsample, split=True, lhs_primer_fwd="", lhs_primer_rev="", rhs_primer_fwd="", rhs_primer_rev="", add_adaptor_through_pcr=True)

    print("Saving probes...")
    designer.save_csv_for_twist(args.output_prefix, final_probe_column='flex_probe_with_primers', split=args.split_probes)

if __name__ == "__main__":
    parsed_args = parse_args(sys.argv[1:])
    main(parsed_args)