# load summaries of simulation results from npz files (generated with summary_stats.py), collate and save to a single csv table
import os
from glob import glob
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
import numpy as np
import pandas as pd
import click

import OBMABCSMC

def distance(real_freq_sum, simulated_freq_sum, real_fit_sum, simulated_fit_sum):
    if simulated_fit_sum.shape[0] == 4: # real data only has transfer 20 and 40
        simulated_fit_sum = simulated_fit_sum[[1,3], :]
    freq_resid = real_freq_sum - simulated_freq_sum
    fit_resid = real_fit_sum - simulated_fit_sum
    rss = (freq_resid**2).sum() + (fit_resid**2).sum()
    return rss


@click.command()
@click.option('--folder',required=True, type=click.Path(file_okay=False, dir_okay=True, readable=True, exists=True))
@click.option('--output-filename', required=True, type=click.Path(file_okay=True, writable=True))
@click.option('--cpus', type=int, default=28)
@click.option('--nfiles', type=int, default=0)
def main(folder, output_filename, cpus, nfiles):
#     folder = 'output_repeat_different_Ub'
#     output_filename = 'output_OBM_single_genotype_repeat_Ub_summary.csv.gz'
    
    files = glob(os.path.join(folder, '*.npz'))
    if nfiles == 0: nfiles = len(files)
    files = files[:nfiles]
    print('{}: processing {} npz files from {}'.format(datetime.now().ctime(), nfiles, folder))

    real_data = OBMABCSMC.load_real_data()
    max_fitness = real_data['Fitness'].max()
    real_freq_sum_Dplus = OBMABCSMC.sum_freq_data(real_data[real_data['type'] == 'dinB+'])
    real_fit_sum_Dplus = OBMABCSMC.sum_fit_data(real_data[real_data['type'] == 'dinB+'], max_fitness=max_fitness)
    real_freq_sum_Dminus = OBMABCSMC.sum_freq_data(real_data[real_data['type'] == 'dinB-'])
    real_fit_sum_Dminus = OBMABCSMC.sum_fit_data(real_data[real_data['type'] == 'dinB-'], max_fitness=max_fitness)

    
    def process_file(fn):
        d = np.load(fn)
        simulated_freq_sum = d['simulated_freq_sum']
        simulated_fit_sum = d['simulated_fit_sum']
        if simulated_fit_sum.shape[0] == 4: # real data only has transfer 20 and 40...
            simulated_fit_sum = simulated_fit_sum[[1,3], :]
        Dplus = distance(real_freq_sum_Dplus, simulated_freq_sum, real_fit_sum_Dplus, simulated_fit_sum)
        Dminus = distance(real_freq_sum_Dminus, simulated_freq_sum, real_fit_sum_Dminus, simulated_fit_sum)
        return dict(Ub=d['Ub'], α=d['α'], β=d['β'], τ=d['τ'], Dplus=Dplus, Dminus=Dminus)


    with ThreadPoolExecutor(cpus) as executor:
        futures = [executor.submit(process_file, fn) for fn in files[:nfiles]]
    
    print('{}: saving results to {}'.format(datetime.now().ctime(), output_filename))
    df = pd.DataFrame([fut.result() for fut in tqdm(as_completed(futures), leave=False)])
    df.to_csv(output_filename, compression='gzip', index=False)
    print('{}: Finished.'.format(datetime.now().ctime()))
    
    
if __name__ == '__main__':
    main()