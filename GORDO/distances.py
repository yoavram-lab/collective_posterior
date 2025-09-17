# load simulation results and calculate summary and distance, saving results to a single csv table
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from itertools import islice
from glob import glob
import sys
import evodyn_gamma_v3
import OBMABCSMC
import pandas as pd

output_filename = 'output_summary.csv.gz'

if __name__ == '__main__':
    real_data = OBMABCSMC.load_real_data()
    real_data_plus = real_data[real_data['type'] == 'dinB-']
    real_data_minus = real_data[real_data['type'] == 'dinB+']
    files = glob('output/*')
    print("Collecting {} output files".format(len(files)))
    
    def f(fn):
        try:
            simulated_data = evodyn_gamma_v3.load(fn, progressbar=False)
        except:
            print("Failed opening file ", fn)
        return dict(
            Ub=simulated_data['Ub'].unique()[0], 
            τ=simulated_data['τ'].unique()[0], 
            α=simulated_data['α'].unique()[0], 
            β=simulated_data['β'].unique()[0],
            Dplus=OBMABCSMC.distance(dict(data=real_data_plus), dict(data=simulated_data)),
            Dminus=OBMABCSMC.distance(dict(data=real_data_minus), dict(data=simulated_data))
        )
    
    with ThreadPoolExecutor() as exec:
        results = exec.map(f, files)
        results = pd.DataFrame(list(results))
    results.to_csv(output_filename, compression='gzip', index=False)
    print("Saved summary to {}".format(output_filename))
