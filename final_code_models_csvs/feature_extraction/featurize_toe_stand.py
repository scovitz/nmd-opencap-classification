from pathlib import Path

import numpy as np
import pandas as pd

from utilsLoaders import read_trc, read_mot
from utils import center_of_mass, angle_between_all


def toe_stand_trc_feats(xyz, markers, fps, com):
    start_win = int(fps*1)
    com_start = com[:start_win,:].mean(0)
    com -= com_start

    rc = xyz[:,np.argmax(markers=='r_calc_study'),:]
    lc = xyz[:,np.argmax(markers=='L_calc_study'),:]
    rc -= rc[:start_win,:].mean(0)
    lc -= lc[:start_win,:].mean(0)

    dt = 1/fps

    int_com_elev = np.sum(np.clip(com[:,1], 0, None)) * dt
    int_mean_heel_elev = np.sum(np.clip((rc[:,1] + lc[:,1])/2, 0, None)) * dt

    # Trunk lean
    c7 = xyz[:,np.argmax(markers=='r_C7'),:]
    trunk = c7 - com.copy()
    grav = np.zeros_like(trunk)
    grav[:,1] = -1
    trunk_lean = angle_between_all(-grav, trunk) * 180/np.pi
    int_trunk_lean = np.sum(trunk_lean) * dt

    return {
            'toe_stand_int_com_elev': float(int_com_elev),
            'toe_stand_int_mean_heel_elev': float(int_mean_heel_elev),
            'toe_stand_int_trunk_lean': float(int_trunk_lean),
           }


def toe_stand_mot_feats(df):
    raa = df['ankle_angle_r'].values
    laa = df['ankle_angle_l'].values

    dt = df.time[1] - df.time[0]
    int_raa = np.sum(raa) * dt
    int_laa = np.sum(laa) * dt
    mean_int_aa = (int_raa + int_laa) / 2

    return {
            'toe_stand_mean_int_aa': float(mean_int_aa),
           }


def feats_toe_stand(trc_fpath, mot_fpath, model_fpath):
    fps, markers, xyz = read_trc(trc_fpath)
    com_xyz = center_of_mass(model_fpath, mot_fpath)

    trc_feats = toe_stand_trc_feats(xyz, markers, fps, com_xyz)

    df = read_mot(mot_fpath)
    mot_feats = toe_stand_mot_feats(df)

    feats = trc_feats.copy()
    feats.update(mot_feats)

    return feats


if __name__ == '__main__':
    feats = feats_toe_stand(snakemake.input['trc'],
                            snakemake.input['mot'],
                            snakemake.input['model'])
    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)


