from pftools.probe_design.util import *
import MERFISH_probe_design.IO.file_io as fio
import MERFISH_probe_design.probe_design.probe_dict as p_d
import MERFISH_probe_design.probe_design.OTTable_dict as ot
import MERFISH_probe_design.probe_design.readout_sequences as rs
import MERFISH_probe_design.probe_design.probe_selection as ps
import MERFISH_probe_design.probe_design.quality_check as qc
import MERFISH_probe_design.probe_design.filters as filters
import MERFISH_probe_design.probe_design.primer_design as primer_design
import multiprocessing as mp

class AbstractProbeDesigner(object):
    """
    Abstract class for probe designers.
    """
    def __init__(self):
        pass

    def design_probes_for_genes(self):
        raise NotImplementedError

    def add_bits_to_probes(self):
        raise NotImplementedError

    def add_primers_to_probes(self):
        raise NotImplementedError

    def add_extra_seqs(self):
        raise NotImplementedError

class CombinatorialProbeDesigner(AbstractProbeDesigner):
    def __init__(self, codebook_path:str, bit_sequences_path:str, transcriptome_path:str,
                trancriptome_ottable_path:str, ncrna_ottable_path:str,
                extra_seq_path:str=None,
                verbose:bool=True,
                n_probes_per_transcript:int=8,
                ncrna_k:int=15,
                mrna_k:int=17,
                offtarget_ncrna_upper_bound:float=0.5,
                offtarget_mrna_upper_bound:float=75, n_threads=-1):

        self._verbose = verbose

        if bit_sequences_path is not None:
            self._bits = pd.read_csv(bit_sequences_path)
        else:
            self._bits = None

        if codebook_path is not None:
            _, _, self._bit_names, self._codebook = fio.load_merlin_codebook(codebook_path)
            self._gene_list = self._codebook['name'].tolist()
            self._gene_list = [g for g in self._gene_list if "Blank" not in g]
        else:
            self._codebook = None

        if self._verbose and codebook_path is not None:
            print(f"Loaded {self._codebook.shape[0]} codes for {len(self._gene_list)} genes.")

        self._transcriptome = pd.read_csv(transcriptome_path)
        if self._verbose:
            print(f"Loaded {self._transcriptome.shape[0]} transcripts.")

        if self._verbose:
            print("Loading OTTables...")
        self._transcriptome_ottable = ot.OTTable.load_pkl(trancriptome_ottable_path)
        self._ncrna_ottable = ot.OTTable.load_pkl(ncrna_ottable_path)
        self._n_probes_per_transcript = n_probes_per_transcript
        self._ncrna_k = ncrna_k
        self._mrna_k = mrna_k
        self._offtarget_ncrna_upper_bound = offtarget_ncrna_upper_bound
        self._offtarget_mrna_upper_bound = offtarget_mrna_upper_bound

        if extra_seq_path is not None:
            self._load_extra_sequences(extra_seq_path)
        else:
            self._extra_seqs = None
            self._extra_seqs_ottable = None
        if n_threads == -1:
            self._n_threads = mp.cpu_count()
        else:
            self._n_threads = n_threads

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

    def _init_probe_dict(self, pb_len:int, overlap:int=1):
        self._probe_dict = p_d.init_probe_dict(self._gene_list, self._transcriptome, 'gene_short_name', K=pb_len, overlap=overlap)
        for gk in self._probe_dict.keys():
            for tk in self._probe_dict[gk].keys():
                curr_df = self._probe_dict[gk][tk]
                self._probe_dict[gk][tk]['barcode'] = [self._codebook[self._codebook['name'] == gk]['barcode_str'].iloc[0]] * curr_df.shape[0]

    def add_primers_to_probes(self, forward_primer_rc:str, reverse_primer:str, subsample_probes:bool=True, input_column:str='target_sequence', output_column:str='target_sequence_with_primers'):
        # remove genes that don't have any designed probes
        new_pb_dict = {}
        for gk in self._probe_dict.keys():
            for tk in self._probe_dict[gk].keys():
                curr_df = self._probe_dict[gk][tk]
                if curr_df.shape[0] > 0:
                    if gk not in new_pb_dict:
                        new_pb_dict[gk] = {}
                    new_pb_dict[gk][tk] = curr_df
                else:
                    print("Found 0 probes for gene", gk, "transcript", tk, ". Removing gene from probe dict.")

        self._probe_dict = new_pb_dict

        primer_design.add_primer_sequences(self._probe_dict, forward_primer_rc, reverse_primer,
                                           input_column=input_column, output_column=output_column)
        # filter out probes that now bind somewhere
        ot.calc_OTs(self._probe_dict, self._ncrna_ottable, output_column, f'{output_column}_OT_ncrna', self._ncrna_k)
        filters.filter_probe_dict_by_metric(self._probe_dict, f'{output_column}_OT_ncrna', upper_bound=self._offtarget_ncrna_upper_bound)

        if subsample_probes:
            print(f"SUBSAMPLING PROBES TO {self._n_probes_per_transcript} PER TRANSCRIPT")
            select_probes_greedy_stochastic_nobits(self._probe_dict, self._n_probes_per_transcript)

    def save_csv_for_twist(self, output_prefix:str, final_probe_column:str='target_sequence_with_primers', pool_name:str=None, split:bool=False):

        # save the probe dict as a csv file
        df = p_d.probe_dict_to_df(self._probe_dict)
        if split:
            lhs_probes, rhs_probes = [], []
            for s in df[final_probe_column].tolist():
                lhs, rhs = s.split(".")
                lhs_probes.append(lhs)
                rhs_probes.append(rhs)
            df['lhs_probe'] = lhs_probes
            df['rhs_probe'] = rhs_probes

        df.to_csv(output_prefix + "_probes.csv", index=False)

        if pool_name is None:
            pool_name = output_prefix

        # save the probe dict as a csv file for Twist
        seqs = df[final_probe_column].tolist()
        if split:
            split_seqs = []
            for s in seqs:
                for r in s.split("."):
                    split_seqs.append(r)
            seqs = split_seqs

        pd.DataFrame({
            'name': [pool_name] * len(seqs),
            'sequence': seqs
        }).to_csv(output_prefix + "_twist.csv", index=False)

        # identify genes that did not make it into the final library
        genes_in_library = set(df['gene_id'].tolist())
        #if self._codebook is not None:
        #    genes_not_in_library = set(self._codebook['id'].tolist()) - genes_in_library
        #    genes_not_in_library = [i for i in genes_not_in_library if "Blank" not in i]
        #    print("Genes not in library:", genes_not_in_library)
        #    pd.DataFrame({'gene_symbol': list(genes_not_in_library)}).to_csv(output_prefix + "_failed.csv", index=False)
        #else:
            # identify genes that did not make it into the final library
        genes_not_in_library = set(self._gene_list) - genes_in_library
        print("Genes not in library:", genes_not_in_library)
        pd.DataFrame({'gene_symbol': list(genes_not_in_library)}).to_csv(output_prefix + "_failed.csv", index=False)

    def get_probes_as_df(self) -> pd.DataFrame:
        df = p_d.probe_dict_to_df(self._probe_dict)
        return df
