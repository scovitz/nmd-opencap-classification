from pathlib import Path
import pandas as pd


def agg(datadir, featdir, *fpaths):
    dfs = []
    for fpath in fpaths:
        ID, day, activity = Path(fpath).stem.split('__')
        df = pd.read_csv(fpath, header=None, names=['feature', 'value'])
        df['ID'] = ID
        df['day'] = int(day)
        dfs.append(df)

    df_feats = pd.concat(dfs)
    df_feats = df_feats.pivot(index=['ID', 'day'], columns='feature', values='value')
    df_feats = df_feats.reset_index()

    df_info = pd.read_csv(Path(datadir) / 'participant_info.csv')
    df_info.sort_values(by=['ID', 'day'], inplace=True)
    df_feats.sort_values(by=['ID', 'day'], inplace=True)
    assert df_info[['ID', 'day']].equals(df_feats[['ID', 'day']])

    # apply normalization to subset of features
    df_feats['10mwt_stride_len'] /= df_info['height']
    df_feats['10mwt_ankle_elev'] /= df_info['height']
    df_feats['10mwrt_stride_len'] /= df_info['height']
    df_feats['10mwrt_ankle_elev'] /= df_info['height']
    df_feats['toe_stand_int_com_elev'] /= df_info['height']
    df_feats['toe_stand_int_mean_heel_elev'] /= df_info['height']
    df_feats['5xsts_stance_width'] /= df_info['height']
    df_feats['arm_rom_rw_area'] /= df_info['height'] ** 2

    # fill missing features with zeros
    df_feats.fillna(0, inplace=True)

    return df_feats


if __name__ == '__main__':
    df_feats = agg(*snakemake.input)
    df_feats.to_csv(snakemake.output[0], index=False)

