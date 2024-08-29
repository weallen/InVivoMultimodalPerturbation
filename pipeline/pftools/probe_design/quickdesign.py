# functions to quickly design probes against sequences
from MERFISH_probe_design.probe_design.OTTable_dict import OTTable

from MERFISH_probe_design.probe_design.filters import calc_tm_JM, GC
from pftools.probe_design.probe_design_paths import *
from pftools.probe_design.util import seqrc

class QuickProbeDesigner:
    def __init__(self, species='mouse'):
        self.species = species
        self.ottable, self.ottable_ncrna = self.load_ottables(species)

    def load_ottables(self, species='mouse'):
        if species == "mouse":
            return OTTable.load_pkl(MOUSE_OTTABLE_17), OTTable.load_pkl(MOUSE_OTTABLE_NCRNA_15) 
        elif species == "human":
            return OTTable.load_pkl(HUMAN_OTTABLE_17), OTTable.load_pkl(HUMAN_OTTABLE_NCRNA_15)
        else:
            raise ValueError("Species not supported.")

    def design_probes(self, seq, bit="", linker='A', overlap=10, 
                      pb_len=30, ncrna_thresh=0, mrna_thresh=50, gc_range=[25, 75], tm_range=[66, 76]):
        #pbs = [i:(i+pb_len) for i in range(0, len(seq)-pb_len, overlap)]
        if len(seq) <= pb_len:
            pbs = [seq]
        else:
            pbs = [seq[i:i+pb_len] for i in range(0, len(seq)-pb_len, overlap)]
        good_pbs = []
        for pb in pbs:
            pb = pb.upper()
            if compute_pb_score(self.ottable, pb, 17) <= mrna_thresh and compute_pb_score(self.ottable_ncrna, pb, 15) <= ncrna_thresh:
                if GC(pb) >= gc_range[0] and GC(pb) <= gc_range[1]:
                    if calc_tm_JM(pb) >= tm_range[0] and calc_tm_JM(pb) <= tm_range[1]:
                        good_pbs.append(seqrc(pb) + linker + bit)
        return good_pbs


def compute_pb_score(ot, pb, k):
    return sum([ot[pb[i:(i+k)]] for i in range(0, len(pb)-k, k)])