import numpy as np
import pandas as pd

def format_arr_for_dataorg(arr:np.ndarray) -> str:
    return "[" + "".join([str(i) + " " for i in arr[:-1]]) + str(arr[-1]) + "]"

def generate_dataorg_for_threecolor_experiment(n_bits:int, n_z:int, n_cols:int=5, 
                                               image_type:str='zscan_5cols_slow', 
                                               image_reg_exp:str='(?P<imageType>[\w|-]+)_(?P<fov>[0-9]+)_(?P<imagingRound>[0-9]+)',
                                               correct_750_zoffset:bool=False,
                                               add_dapi_channel:bool=True,
                                               z_spacing:float=1.,
                                               start_idx:int=0,
                                               swap_dapi:bool=False) -> pd.DataFrame:                               

    """
    Generate data organization for an experiment that uses 3 color imaging with 3 bits per round.
    Assume that the bits have a cyclic color organization.
    """                
    chan_names = []
    color = []
    frame = []
    z_pos = []
    readout_name = []
    imaging_round = []
    idx = np.arange(0, n_z*n_cols, n_cols)
    round_idx = 0
    for i in range(n_bits):
        chan_names.append(f"Bit{i+1}")
        readout_name.append(f"R{i+1}")
        imaging_round.append(round_idx)
        if i%3 == 0:
            color.append("560")
            if correct_750_zoffset:
                # truncate last z plane
                curr_idx = idx[:-1]
            else:
                curr_idx = idx
            if swap_dapi:
                frame.append(format_arr_for_dataorg(curr_idx+3))
            else:
                frame.append(format_arr_for_dataorg(curr_idx+2))
        elif i%3 == 1:
            color.append("650")
            if correct_750_zoffset:
                curr_idx = idx[:-1]
            else:
                curr_idx = idx
            if swap_dapi:
                frame.append(format_arr_for_dataorg(curr_idx+2))
            else:
                frame.append(format_arr_for_dataorg(curr_idx+1))
        elif i%3 == 2:
            color.append("750")
            if correct_750_zoffset:
                curr_idx = idx[1:]
            else:
                curr_idx = idx
            if swap_dapi:
                frame.append(format_arr_for_dataorg(curr_idx+1))
            else:
                frame.append(format_arr_for_dataorg(curr_idx))
            round_idx += 1
        if correct_750_zoffset:
            z_pos.append(z_spacing*np.arange(n_z-1))
        else:
            z_pos.append(z_spacing*np.arange(n_z))
    if add_dapi_channel:
        if correct_750_zoffset:
            curr_idx = idx[:-1]
        else:
            curr_idx = idx
        chan_names.append("DAPI")
        readout_name.append("DAPI")
        imaging_round.append(0) # use first round
        color.append("405")
        if swap_dapi:
            frame.append(format_arr_for_dataorg(curr_idx))
        else:
            frame.append(format_arr_for_dataorg(curr_idx+n_cols-1))
        if correct_750_zoffset:
            z_pos.append(z_spacing*np.arange(n_z-1))
        else:
            z_pos.append(z_spacing*np.arange(n_z))
    fiducial_frame = []
    for i in range(len(imaging_round)):
        if swap_dapi:
            fiducial_frame.append(0)
        else:
            fiducial_frame.append(n_cols-1)
    return pd.DataFrame({'channelName': chan_names,
                        'readoutName': readout_name,
                        'imageType': [image_type]*len(chan_names),
                        'imageRegExp':[image_reg_exp]*len(chan_names),
   #                     'bitNumber':np.arange(n_bits)+1,
                        'color':color,
                        'frame':frame,
                        'zPos': z_pos,
                        'imagingRound': imaging_round,
                        'fiducialImageType':image_type,
                        'fiducialRegExp':image_reg_exp,
                        'fiducialImagingRound':imaging_round,
                        'fiducialFrame':fiducial_frame,
                        'fiducialColor':['405']*len(imaging_round)})