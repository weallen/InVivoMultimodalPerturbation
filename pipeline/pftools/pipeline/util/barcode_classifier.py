import numpy as np
import xgboost as xgb
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm
from pftools.pipeline.core.codebook import Codebook
from scipy.optimize import newton
import multiprocessing

def featurize_molecules(mols:pd.DataFrame, codebook:Codebook) -> pd.DataFrame:
    colnames = mols.columns
    intensity_cols = [i for i in colnames if i.startswith("intensity_")]
    inten_values = mols.loc[:,intensity_cols].values

    on_bits, off_bits = get_on_off_bit_intensities_for_molecules(mols, codebook)
    feats = mols.loc[:,["min_distance","max_intensity", "mean_intensity","area","mean_distance"]]# + intensity_cols]
    feats['min_distance'] = feats['min_distance'].astype(np.float32)
    feats['max_intensity'] = feats['max_intensity'].astype(np.float32)
    feats['mean_intensity'] = feats['mean_intensity'].astype(np.float32)
    feats['area'] = feats['area'].astype(np.float32)
    feats['mean_distance'] = feats['mean_distance'].astype(np.float32)
    feats = feats.assign(max_inten = np.max(inten_values, axis=1).astype(np.float32),
                        min_inten = np.min(inten_values, axis=1).astype(np.float32),
                        var_inten = np.std(inten_values, axis=1).astype(np.float32),
                        diff_inten = (np.max(inten_values, axis=1) - np.min(inten_values, axis=1)).astype(np.float32),
                        snr_inten = (np.mean(inten_values, axis=1)/np.std(inten_values, axis=1)).astype(np.float32),
                        mean_on = np.mean(on_bits, axis=1).astype(np.float32),
                        mean_off = np.mean(off_bits, axis=1).astype(np.float32),
                        var_on = np.std(on_bits,axis=1).astype(np.float32),
                        var_off = np.std(off_bits,axis=1).astype(np.float32),
                        min_on = np.min(on_bits,axis=1).astype(np.float32),
                        max_on = np.max(on_bits,axis=1).astype(np.float32),
                        min_off = np.min(off_bits,axis=1).astype(np.float32),
                        max_off = np.max(off_bits,axis=1).astype(np.float32),
                        snr_on = (np.mean(on_bits,axis=1)/np.std(on_bits,axis=1)).astype(np.float32),
                        snr_off = (np.mean(off_bits,axis=1)/np.std(off_bits,axis=1)).astype(np.float32),
                        snr = (np.mean(on_bits,axis=1)/np.mean(off_bits,axis=1)).astype(np.float32))
    feats.replace([np.inf, -np.inf], 0, inplace=True)
    return feats

def get_on_off_bit_intensities_for_molecules(mols:pd.DataFrame, codebook:Codebook) -> Tuple[np.ndarray, np.ndarray]:
    # get intensity matrix
    colnames = mols.columns
    intensity_cols = [i for i in colnames if "intensity_" in i]
    intensities = mols.loc[:,intensity_cols].values.astype(np.float32)
    codes = codebook.get_barcodes()
    # get barcodes (assumes in same order as codebook)
    bc = np.array(mols.barcode_id)
    on_bits = []
    off_bits = []
    for i,b in enumerate(bc):
        # get bits for barcode
        bits = codes[b,:]
        on_bit_vals = intensities[i,bits==1].astype(np.float32)
        off_bit_vals = intensities[i,bits==0].astype(np.float32)
        on_bits.append(on_bit_vals)
        off_bits.append(off_bit_vals)
    return np.vstack(on_bits).astype(np.float32), np.vstack(off_bits).astype(np.float32)

def predict_molecule_quality(mols:pd.DataFrame, codebook:Codebook, n_repeats:int=5, random_state:int=42, max_molecules:int=1000000) -> xgb.XGBClassifier:
    """
    Train an XGBoost classifier on the given molecules and codebook
    """
    # get coding and noncoding features
    np.random.seed(random_state)
    print("Featurizing molecules")
    feats = pd.concat([featurize_molecules(mols[i:i+max_molecules], codebook) for i in tqdm(range(0, mols.shape[0], max_molecules))], ignore_index=True)
    coding_feats = feats[mols.barcode_id.isin(codebook.get_coding_indexes())]
    noncoding_feats = feats[mols.barcode_id.isin(codebook.get_blank_indexes())]
    yprobs = []
    mols = mols.assign(is_coding = mols.barcode_id.isin(codebook.get_coding_indexes()))
    print("Classifying molecules")
    for i in tqdm(range(n_repeats)):
        noncoding_feats_temp = noncoding_feats.sample(min(len(noncoding_feats), max_molecules))
        coding_feats_temp = coding_feats.sample(len(noncoding_feats_temp))
        X = np.vstack([coding_feats_temp.values, noncoding_feats_temp.values]).astype(np.float32)
        y = np.hstack([np.ones((len(coding_feats_temp),)), np.zeros((len(noncoding_feats_temp),))]).astype(np.float32)
        #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        model = xgb.XGBClassifier(tree_method="hist", max_depth=10, objective='binary:logistic',nthread=multiprocessing.cpu_count()-1)
        model.fit(X, y)
        #y_pred = model.predict(X_test)
        #accuracy = accuracy_score(y_test, y_pred)
        #auc = roc_auc_score(y_test, y_pred)
        #print("Accuracy: %.2f%%" % (accuracy * 100.0))
        #print("AUC: %.2f%%" % (auc * 100.0))
        all_yprob = np.hstack([model.predict_proba(feats.values[i:i+max_molecules])[:,0] for i in range(0, feats.shape[0], max_molecules)])
        yprobs.append(all_yprob)
    mols = mols.assign(quality = np.mean(yprobs, axis=0))
    return mols

def calculate_misidentification_rate_for_threshold(mols:pd.DataFrame, n_coding:int, n_blank:int, threshold:float) -> float:
    """
    Calculate the normalized misidentification rate for a given threshold
    """
    mols_filt = mols[mols.quality < threshold]
    blank_counts = np.sum(~mols_filt.is_coding)
    coding_counts = np.sum(mols_filt.is_coding)
    return (blank_counts/n_blank)/(coding_counts/n_coding)

def calculate_threshold_for_misidentification_rate_ml(mols:pd.DataFrame, codebook:Codebook, target_rate:float=0.05, tol:float=0.0001) -> float:
    """
    Calculate the threshold for quality score that gives a particular misidentification rate.
    """
    n_coding = len(codebook.get_coding_indexes())
    n_blank = len(codebook.get_blank_indexes())

    def misidentification_rate(threshold:float) -> float:
        return calculate_misidentification_rate_for_threshold(mols, n_coding, n_blank, threshold) - target_rate
    if misidentification_rate(0.5) < target_rate:
        return 0.5
    else:
        return newton(misidentification_rate, 0.5, tol=tol, disp=False)
