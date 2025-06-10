
from pathlib import Path
import pandas as pd
from featurize_gait import featurize_gait


if __name__ == '__main__':
    feats = featurize_gait(snakemake.input['trc'],
                           snakemake.input['mot'],
                           snakemake.input['model'],
                           '10mwt'
                          )

    outpath = Path(snakemake.output[0])
    outpath.parent.mkdir(exist_ok=True)
    df = pd.DataFrame.from_dict(feats, orient='index')
    df.to_csv(outpath, header=False)


