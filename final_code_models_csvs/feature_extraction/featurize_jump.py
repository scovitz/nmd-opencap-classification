from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal as ss

from utilsLoaders import read_trc
from utils import center_of_mass_vel


def jump_trc_feats(comv, fps):
    # LP filter kernel
    win = ss.windows.hann(int(0.5*fps))
    win /= np.sum(win)

    comv_y = ss.convolve(comv[:,1], win, mode='same')
    max_com_vel = np.max(comv_y)

    return {
            'jump_max_com_vel': float(max_com_vel),
           }


def feats_jump(trc_fpath, mot_fpath, model_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    comv = center_of_mass_vel(model_fpath, mot_fpath)
    feats = jump_trc_feats(comv, fps)
    return feats


if __name__ == '__main__':
    feats = feats_jump(snakemake.input['trc'],
                       snakemake.input['mot'],
                       snakemake.input['model'])
    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)


