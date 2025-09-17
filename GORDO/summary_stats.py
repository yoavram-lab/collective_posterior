# load simulation results and calculate summaries, saving to npz files, one per simulation
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import islice
import os
from glob import glob
import sys
import evodyn_gamma_v3
import OBMABCSMC
import pandas as pd
import numpy as np
from datetime import datetime
import click
from tqdm import tqdm


@click.command()
@click.option('--folder', default='output', type=click.Path(readable=True, exists=True, dir_okay=True, file_okay=False))
def main(folder):
    real_data = OBMABCSMC.load_real_data()
    max_fitness = real_data['Fitness'].max()

    files = glob(os.path.join(folder, '*'))
    print("{}: Collecting {} output files".format(datetime.now().ctime(), len(files)))
    
    def summarize(fn):
        try:
            simulated_data = evodyn_gamma_v3.load(fn, progressbar=False)
        except:
            print("Failed opening file ", fn)
        simulated_freq_sum = OBMABCSMC.sum_freq_data(simulated_data)
        simulated_fit_sum = OBMABCSMC.sum_fit_data(simulated_data, max_fitness=max_fitness)
        np.savez_compressed(fn.replace('.txt', '.summary.npz'),
            Ub=simulated_data['Ub'].unique()[0], 
            τ=simulated_data['τ'].unique()[0], 
            α=simulated_data['α'].unique()[0], 
            β=simulated_data['β'].unique()[0],
            simulated_freq_sum=simulated_freq_sum,
            simulated_fit_sum=simulated_fit_sum
        )

    with ThreadPoolExecutor() as exec:
        futures = [exec.submit(summarize, file) for file in files]
        
        kwargs = {
            'total': len(futures),
            'unit': 'simulation',
            'leave': False
        }

        for f in tqdm(as_completed(futures), **kwargs):
            if f.exception():
                click.echo(f.exception())
                
    print("{}: Finished.".format(datetime.now().ctime()))



if __name__ == '__main__':
    main()
