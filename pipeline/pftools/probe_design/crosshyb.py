from Bio.Seq import Seq
from Bio.SeqUtils.MeltingTemp import Tm_NN, chem_correction, DNA_NN4
from nupack import *
from tqdm import tqdm
from joblib import Parallel, delayed
import numpy as np
from pftools.probe_design.util import seqrc

def ctm(s, Na=300, fmd=20, dnac1=1, dnac2=1):
    return chem_correction(Tm_NN(Seq(s),nn_table=DNA_NN4, Na=Na, dnac1=dnac1, dnac2=dnac2),fmd=fmd)

def compute_crosshyb(seq1, seq2, conc1=1e-8, conc2=1e-8, temp=37): # use 10 nM as base concentration
    model = Model(material='dna', celsius=temp, sodium=0.3)
    a = Strand(seq1,name='a')
    b = Strand(seq2,name='b')
    t1 = Tube(strands={a:conc1, b:conc2}, complexes=SetSpec(max_size=2), name='Tube t1')
    tube_result = tube_analysis(tubes=[t1], compute=['pairs'], model=model)
    concs = {c.name : conc for c, conc in tube_result.tubes[t1].complex_concentrations.items()}
    #return concs['(a+b)']
    if '(a+b)' in concs:
        return concs['(a+b)']/conc1
    else:
        return concs['(b+a)']/conc1

def simulate_all_crosshybs(s, temp=37):
    s_ext = s #+ [seqrc(i) for i in s]
    crosshybs = np.zeros((len(s_ext), len(s_ext)))
    for i in tqdm(range(len(s_ext))):
        s1 = s_ext[i]
        crosshybs[i,:] = np.array(Parallel(n_jobs=-1)(delayed(compute_crosshyb)(s1,seqrc(s2), temp=temp) for s2 in s_ext))
    return crosshybs

def simulate_all_crosshybs_with_repeats(s, temp=37, nrep=16):
    s_ext = s #+ [seqrc(i) for i in s]
    crosshybs = np.zeros((len(s_ext), len(s_ext)))
    for i in tqdm(range(len(s_ext))):
        s1 = s_ext[i]
        crosshybs[i,:] = np.array(Parallel(n_jobs=-1)(delayed(compute_crosshyb)(s1 * nrep,seqrc(s2), temp=temp) for s2 in s_ext))
    return crosshybs

def simulate_ensemble(v):
    model = Model(material='dna', celsius=20, sodium=0.3)
    nvars = len(v)
    strands = [Strand(s,name='s'+str(i)) for i,s in enumerate(v)]
    strands_rc = [Strand(seqrc(s),name='s*'+str(i)) for i,s in enumerate(v)]
    strand_names = ['s'+str(i) for i in range(nvars)] + ['s*'+str(i) for i in range(nvars)]
    strands_conc = {s:1e-8 for s in strands+strands_rc}
    t1 = Tube(strands=strands_conc, complexes=SetSpec(max_size=2), name='Tube t1')
    tube_result = tube_analysis(tubes=[t1], compute=['pairs'], model=model)
    concs = {c.name : conc for c, conc in tube_result.tubes[t1].complex_concentrations.items()}
    concs_pairwise = []
    for s1 in range(nvars):
        for s2 in range(nvars):
            curr_name = '(s' + str(s1) +'+s*' + str(s2)+')'
            curr_name_rev = '(s' + str(s2) +'+s*' + str(s1)+')'
            curr_name_rev_swap = '(s' + str(s1) +'+s*' + str(s2)+')'
            curr_name_swap = '(s*' + str(s1) +'+s' + str(s2)+')'
            if curr_name in concs:
                concs_pairwise.append(concs[curr_name]/1e-8)
            elif curr_name_rev in concs:
                concs_pairwise.append(concs[curr_name_rev]/1e-8)
            elif curr_name_rev_swap in concs:
                concs_pairwise.append(concs[curr_name_rev_swap]/1e-8)
            elif curr_name_swap in concs:
                concs_pairwise.append(concs[curr_name_swap]/1e-8)
    return np.array(concs_pairwise).reshape((nvars,nvars))
        