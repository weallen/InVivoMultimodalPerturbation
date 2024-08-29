import numpy as np
import pandas as pd
import os
from typing import List

import MERFISH_probe_design.IO.file_io as fio
import MERFISH_probe_design.probe_design.probe_dict as p_d
import MERFISH_probe_design.probe_design.OTTable_dict as ot
import MERFISH_probe_design.probe_design.readout_sequences as rs
import MERFISH_probe_design.probe_design.probe_selection as ps
import MERFISH_probe_design.probe_design.quality_check as qc
from MERFISH_probe_design.probe_design import filters
from MERFISH_probe_design.probe_design import plot
from MERFISH_probe_design.probe_design import primer_design
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp
from pftools.probe_design.codebook import convert_code_list, stringify_binary_mat_codebook
from pftools.probe_design.base_probe_design import CombinatorialProbeDesigner
from pftools.probe_design.util import *
import multiprocessing as mp

#
# Step 3: Design probes for the transcriptome
#


class MERFISHProbeDesign(CombinatorialProbeDesigner):
    def __init__(self, codebook_path:str, bit_sequences_path:str, transcriptome_path:str,
                 transcriptome_ottable_path:str, ncrna_ottable_path:str,
                 extra_seq_path:str=None,
                 verbose:bool=True,
                 n_probes_per_transcript:int=8,
                 ncrna_k:int=15,
                 mrna_k:int=17,
                 offtarget_ncrna_upper_bound:float=0.5,
                 offtarget_mrna_upper_bound:float=75, n_threads=-1):
        # initialize super class
        super().__init__(codebook_path, bit_sequences_path, transcriptome_path,
                            transcriptome_ottable_path, ncrna_ottable_path,
                            extra_seq_path=extra_seq_path,
                            verbose=verbose,
                            n_probes_per_transcript=n_probes_per_transcript,
                            ncrna_k=ncrna_k,
                            mrna_k=mrna_k,
                            offtarget_ncrna_upper_bound=offtarget_ncrna_upper_bound,
                            offtarget_mrna_upper_bound=offtarget_mrna_upper_bound,
                            n_threads=n_threads)


    def design_probes_for_genes(self, pb_len:int,
                                    gc_range:List[float]=[25, 75],
                                    tm_range:List[float]=[66, 76],
                                    na_conc_molar:float=0.3,
                                    overlap:int=10):
        # Initialize the probe dictionary
        super()._init_probe_dict(pb_len, overlap=overlap)

        # reverse complement the probes
        p_d.get_rc_sequences(self._probe_dict, 'target_sequence', 'target_sequence_rc')

        filter_probe_dict(self._probe_dict, "target_sequence_rc",
                        self._ncrna_ottable, self._transcriptome_ottable,
                        gc_range=gc_range, tm_range=tm_range, ncrna_k=self._ncrna_k,
                        offtarget_ncrna_upper_bound=self._offtarget_ncrna_upper_bound,
                        mrna_k=self._mrna_k, off_target_mrna_upper_bound=self._offtarget_mrna_upper_bound,
                        na_conc_molar=na_conc_molar, write_prefix='target_sequence_rc')

        if self._extra_seqs is not None:
            ot.calc_OTs(self._probe_dict, self._extra_seqs_ottable, 'target_sequence_rc', 'target_sequence_rc_OT_extra', self._mrna_k)
            filters.filter_probe_dict_by_metric(self._probe_dict, 'target_sequence_rc_OT_extra', upper_bound=self._offtarget_mrna_upper_bound)

        # swap target_sequence_rc and target_sequence -- target_sequence_rc is what actually binds
        for gk in self._probe_dict:
            for tk in self._probe_dict[gk]:
                curr_df = self._probe_dict[gk][tk]
                if curr_df.shape[0] > 0:
                    seqs, seqs_rc = curr_df['target_sequence'], curr_df['target_sequence_rc']
                    curr_df['target_sequence'] = seqs_rc
                    curr_df['target_sequence_rc'] = seqs
                    self._probe_dict[gk][tk] = curr_df

    def add_bits_to_probes(self, n_readout_per_probe:int=4, bit_revcomp=True, spacer=''):
        # Each probe will get all bits in the codebook
        readout_seqs = self._bits
        readout_seqs = readout_seqs[readout_seqs['bit_name'].isin(self._bit_names)]
        if bit_revcomp:
            readout_seqs['bit_sequence'] = readout_seqs['bit_sequence'].apply(seqrc)
        readout_seqs['on-bit'] = np.arange(readout_seqs.shape[0])
        readout_seqs['sequence'] = readout_seqs['bit_sequence'] # rename for add_readout_seqs_to_probes_random

        rs.add_readout_seqs_to_probes_random(self._probe_dict, readout_seqs, self._codebook, N_readout_per_probe=n_readout_per_probe, spacer=spacer, n_threads=self._n_threads)
        ot.calc_OTs(self._probe_dict, self._ncrna_ottable, 'target_readout_sequence', 'target_readout_OT_rtRNA', self._ncrna_k)
        filters.filter_probe_dict_by_metric(self._probe_dict, 'target_readout_OT_rtRNA', upper_bound=0.5)


    def add_primers_to_probes(self, forward_primer_rc:str, reverse_primer:str, subsample_probes:bool=True):
        # select the probes
        if subsample_probes:
            print("Subsampling probes...")
            n_on_bits = np.sum([1 if i == '1' else 0 for i in str(self._codebook['barcode_str'].iloc[0])])  # assume all codes have the same number of on bits
            ps.select_probes_greedy_stochastic(self._probe_dict, N_probes_per_transcript=self._n_probes_per_transcript, N_on_bits=n_on_bits, N_threads=self._n_threads)

        primer_design.add_primer_sequences(self._probe_dict, forward_primer_rc, reverse_primer, input_column='target_readout_sequence', output_column='target_readout_primer_sequence')
        while True:
            # Make a OTTable from the reverse-complement sequences of the probes.
            ottable_probes_rc = ot.get_OTTable_for_probe_dictionary(self._probe_dict, 'target_readout_primer_sequence', 15, rc=True)

            # The off-targets in this table indicates cis/trans-complementarity
            ot.calc_OTs(self._probe_dict, ottable_probes_rc, 'target_readout_primer_sequence', 'probe_cis_trans_OT', 15)
            max_ot = max(plot.get_values_from_probe_dict(self._probe_dict, 'probe_cis_trans_OT'))
            if max_ot == 0:
                break

            # Remove probes that have any cis/trans-complementarity
            filters.filter_probe_dict_by_metric(self._probe_dict, 'probe_cis_trans_OT', upper_bound=max_ot - 0.5)


    def save_csv_for_twist(self, output_prefix: str, final_probe_column: str = 'target_readout_primer_sequence', pool_name: str = None):
        super().save_csv_for_twist(output_prefix, final_probe_column, pool_name)
        transcript_level_report = qc.generate_transcript_level_report(self._probe_dict, self._transcriptome)
        transcript_level_report.to_csv(output_prefix + "_report.csv", index=False)
