from typing import Dict, List, Tuple, Optional
import pandas as pd
import os
import numpy as np

import MERFISH_probe_design.IO.file_io as fio
import MERFISH_probe_design.probe_design.probe_dict as p_d
import MERFISH_probe_design.probe_design.OTTable_dict as ot
import MERFISH_probe_design.probe_design.readout_sequences as rs
import MERFISH_probe_design.probe_design.probe_selection as ps
import MERFISH_probe_design.probe_design.quality_check as qc
from MERFISH_probe_design.probe_design import filters
from MERFISH_probe_design.probe_design import primer_design
from pftools.probe_design.util import select_probes_greedy_stochastic_nobits, seqrc, filter_probe_dict
from pftools.probe_design.base_probe_design import AbstractProbeDesigner

class SMFishProbeDesigner(AbstractProbeDesigner):
    def __init__(self, gene_list_path:str, bit_sequences_path:str, transcriptome_path:str, transcriptome_ottable_path:str, ncrna_ottable_path:str, verbose:bool=True, n_probes_per_transcript:int=48,
                 ncrna_k:int=15, 
                 extra_seq_path:str=None,
                 offtarget_ncrna_upper_bound:float=0.5,
                 mrna_k:int=17,
                 offtarget_mrna_upper_bound:float=75):
        self._verbose = verbose

        self._gene_list = pd.read_csv(gene_list_path, names=['gene_symbol', 'bit_name'])
        if self._verbose:
            print(f"Loaded {self._gene_list.shape[0]} genes.")

        if self._verbose:
            print("Loading transcriptome...")
        self._transcriptome = pd.read_csv(transcriptome_path)
        if self._verbose:
            print(f"Loaded {self._transcriptome.shape[0]} transcripts.")

        if self._verbose:
            print("Loading OTTables...")

        self._transcriptome_ottable = ot.OTTable.load_pkl(transcriptome_ottable_path)
        self._ncrna_ottable = ot.OTTable.load_pkl(ncrna_ottable_path)
        self._n_probes_per_transcript = n_probes_per_transcript
        self._bits = pd.read_csv(bit_sequences_path, names=['bit_name', 'bit_sequence'])

        self._ncrna_k = ncrna_k
        self._mrna_k = mrna_k
        self._offtarget_ncrna_upper_bound = offtarget_ncrna_upper_bound
        self._offtarget_mrna_upper_bound = offtarget_mrna_upper_bound
        if extra_seq_path is not None:
            self._load_extra_sequences(extra_seq_path)
        else:
            self._extra_seqs = None
            self._extra_seqs_ottable = None

    def _load_extra_sequences(self, extra_seq_path:str):
        # this loads as a dataframe with columns 'id', 'description', 'sequence'
        self._extra_seqs = fio.load_fasta_into_df(extra_seq_path)
        print("Loaded", self._extra_seqs.shape[0], "extra sequences.")
        # convert extra seqs to transcriptome dataframe
        self._extra_seqs['gene_short_name'] = self._extra_seqs['id']
        self._extra_seqs['FPKM'] = [1] * self._extra_seqs.shape[0]
        self._extra_seqs['transcript_id'] = self._extra_seqs['id']
        self._extra_seqs['gene_id'] = self._extra_seqs['id']
        self._extra_seqs = self._extra_seqs[['gene_short_name', 'FPKM', 'transcript_id', 'gene_id', 'sequence']]
        # append extra seqs to the transcriptome dataframe
        self._transcriptome = pd.concat([self._transcriptome, self._extra_seqs])

        # get the OTTable for the extra sequences
        self._extra_seqs_ottable = ot.get_OTTable_for_transcriptome(self._extra_seqs, K=self._mrna_k)

    def design_probes_for_genes(self, pb_len:int, 
                                       gc_range:List[float]=[25, 75], 
                                       tm_range:List[float]=[66, 76], 
                                       na_conc_molar:float=0.3):
        # Initialize the probe dictionary
        self._probe_dict = p_d.init_probe_dict(self._gene_list['gene_symbol'].tolist(), self._transcriptome, 'gene_short_name', K=pb_len, overlap=10)

        # Filter the probe dict
        filter_probe_dict(self._probe_dict, "target_sequence", 
                          self._ncrna_ottable, self._transcriptome_ottable,
                          gc_range=gc_range, 
                          tm_range=tm_range, 
                          ncrna_k=self._ncrna_k, 
                          offtarget_ncrna_upper_bound=self._offtarget_ncrna_upper_bound, 
                          mrna_k=self._mrna_k, 
                          off_target_mrna_upper_bound=self._offtarget_mrna_upper_bound, 
                          na_conc_molar=na_conc_molar)


        # Add reverse complement
        p_d.get_rc_sequences(self._probe_dict, 'target_sequence', 'target_sequence_rc')

        if self._extra_seqs is not None:
            ot.calc_OTs(self._probe_dict, self._extra_seqs_ottable, 'target_sequence_rc', 'target_sequence_rc_OT_extra', self._mrna_k)
            filters.filter_probe_dict_by_metric(self._probe_dict, 'target_sequence_rc_OT_extra', upper_bound=self._offtarget_mrna_upper_bound)

    def add_bits_to_probes(self, bit_revcomp=True, add_three_prime_bit=True, spacer='', n_repeats:int=1):
        # Add the bits to the probe dictionary

        # iterate over genes
        for gk in self._probe_dict.keys():
            # get bits
            bit_name = self._gene_list.loc[self._gene_list['gene_symbol'] == gk, 'bit_name'].iloc[0]
            bit_sequence = self._bits.loc[self._bits['bit_name'] == bit_name, 'bit_sequence'].iloc[0]
            
            # get the current bit
            if bit_revcomp:
                bit_sequence = seqrc(bit_sequence)

            # iterate over transcripts
            for tk in self._probe_dict[gk].keys():
                curr_pb_df = self._probe_dict[gk][tk]
                for i, row in curr_pb_df.iterrows():
                    curr_pb_df.at[i, 'bit_sequence'] = bit_sequence 
                    curr_pb_df.at[i, 'bit_name'] = bit_name

                    # Append the bit sequence to the target sequence and save in target_readout_sequence
                    if add_three_prime_bit:
                        curr_pb_df.at[i, 'target_readout_sequence'] = row['target_sequence_rc'] + (spacer + bit_sequence)*n_repeats
                    else:
                        curr_pb_df.at[i, 'target_readout_sequence'] = (bit_sequence + spacer)*n_repeats + row['target_sequence_rc']
            self._probe_dict[gk][tk] = curr_pb_df

        ot.calc_OTs(self._probe_dict, self._ncrna_ottable, 'target_readout_sequence', 'target_readout_OT_ncRNA', self._ncrna_k)
        filters.filter_probe_dict_by_metric(self._probe_dict, 'target_readout_OT_ncRNA', upper_bound=self._offtarget_ncrna_upper_bound)
        select_probes_greedy_stochastic_nobits(self._probe_dict, self._n_probes_per_transcript)

    def add_primers_to_probes(self,  forward_primer_rc:str, reverse_primer:str,subsample_probes:bool=True):
        # Add the primers to the probe dictionary
        primer_design.add_primer_sequences(self._probe_dict, forward_primer_rc, reverse_primer, 
                                           input_column='target_readout_sequence', output_column='target_readout_sequence_with_primer') 
        # filter out probes that now bind somewhere
        ot.calc_OTs(self._probe_dict, self._ncrna_ottable, 'target_readout_sequence_with_primer', 'target_readout_sequence_with_primer_OT_ncrna', self._ncrna_k)
        filters.filter_probe_dict_by_metric(self._probe_dict, 'target_readout_sequence_with_primer_OT_ncrna', upper_bound=self._offtarget_ncrna_upper_bound)

        if subsample_probes:
            select_probes_greedy_stochastic_nobits(self._probe_dict, self._n_probes_per_transcript)

    def save_as_96well_plate(self, output_path:str, final_column:str='target_readout_sequence'):

        self._probe_table = p_d.probe_dict_to_df(self._probe_dict)
        # Generate the indices of a 96 well plates with 8x12 wells, where the first well is A1
        well_indices = []
        for i in range(8):
            for j in range(12):
                well_indices.append(chr(ord('A') + i) + str(j+1))

        # Save the probe dictionary as a 96 well plate, with columns Well Position, Name, and Sequence.
        # If there are more than 96 probes, split into multiple 96 well plates as separate excel sheets.
        n_plates = int(np.ceil(self._probe_table.shape[0] / 96))
        with pd.ExcelWriter(output_path) as writer:
            for i in range(n_plates):
                pt = self._probe_table.iloc[i*96:(i+1)*96, :]
                pt.loc[:,'Well Position'] = well_indices[:pt.shape[0]]
                pt.loc[:,'Name'] = pt['gene_id']
                pt.loc[:,'Sequence'] = pt[final_column]
                pt[['Well Position', 'Name', 'Sequence']].to_excel(writer, sheet_name=f'Plate {i+1}', index=False)

    def save_as_opool_excel(self, pool_name, output_path:str, final_column:str='target_readout_sequence'):

        self._probe_table = p_d.probe_dict_to_df(self._probe_dict)
        # Save the probe dictionary as an Excel file
        pd.DataFrame({'Pool name': [pool_name]*self._probe_table.shape[0], # for some reason IDT wants name to be lower case
                      "Sequence": self._probe_table[final_column]}).to_excel(output_path, index=False)