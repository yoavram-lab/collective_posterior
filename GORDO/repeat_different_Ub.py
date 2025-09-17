# read params from summary table and run again with same alpha beta tau but different Ub
import numpy as np
import pandas as pd
from datetime import datetime
import evodyn_gamma_v3
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

input_file = 'output_OBM_single_genotype_summary.csv.gz'

@click.command()
# @click.option('-v/-V', '--verbose/--no-verbose', default=False)
def main()
    now = datetime.now().ctime()
    print(f"{now}: Reading parameter sets from '{input_file}'")

    df = pd.read_csv(input_file, dtype=float)

    DILUTION = 1/10000
    GEN_PER_DAY = np.log2(1/DILUTION)

    Ubs = 10**np.random.uniform(-9, -4, size=df.shape[0])
    alphas = df['α']
    betas = df['β']
    N0 = int(1.2e4)
    DILUTION = 1/10000
    GEN_PER_DAY = np.log2(1/DILUTION)
    bottleneck = int(GEN_PER_DAY)
    generations = 550
    repeats = 100
    interval = 10
    tau = 1

    def _simulate(i, row):
        params = row.to_dict()
        params['Ub'] = Ubs[i]
        params['alpha'] = params.pop('α')
        params['beta'] = params.pop('β')
        params['tau'] = params.pop('τ')
        return evodyn_gamma_v3.simulate(N0=N0, bottleneck=bottleneck, generations=generations, repeats=repeats, interval=interval, seed=i, **params)       
        
    with ThreadPoolExecutor(28*2) as exec:
        futures = [exec.submit(_simulate, i, row) for i, row in df[['α', 'β', 'τ']].iterrows()]        
        kwargs = {
            'total': len(futures),
            'unit': 'simulation',
            'leave': False
        }

        for f in tqdm(as_completed(futures), **kwargs):
            if f.exception():
                print(f.exception())
        
    print(f'{now}: Finished.')

    
if __name__ == '__main__':
    main()