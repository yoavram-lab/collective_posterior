import numpy as np
import seaborn as sns
import pandas as pd
import os

from pyabc import ABCSMC, RV, Distribution, History
from pyabc.visualization import plot_kde_matrix, plot_kde_1d, plot_kde_2d

import sys
sys.path.append('../src/')
import evodyn_gamma_v3

DILUTION = 1/10000
GEN_PER_DAY = np.log2(1/DILUTION)

data_folder = '../data/13NOV2018/'

def load_real_data():
    data = pd.read_table(data_folder+'24_single_genotype.txt', comment='#')
    data.dropna(axis=1, inplace=True)
    freq_data = pd.melt(data, id_vars=['Population', 'type'], value_vars=['f.0', 'f.10', 'f.20', 'f.30', 'f.40'],
                   var_name='Transfer', value_name='Frequency')
    fit_data = pd.melt(data, id_vars=['Population', 'type'], value_vars=['fit.20', 'fit.40'],
                   var_name='Transfer', value_name='Fitness')

    for data in (freq_data, fit_data):
        data['Transfer'] = data['Transfer'].str.split('.', expand=True)[1]
        data['Transfer'] = data['Transfer'].astype('int')


    # dinB+ ('P' type), the second twelve are dinB- ('M' type).
    for data in (freq_data, fit_data):
        data.loc[data['type'] == 'P', 'type'] = 'dinB+'
        data.loc[data['type'] == 'M', 'type'] = 'dinB-'

    real_data = pd.merge(freq_data, fit_data, on=['Population', 'Transfer', 'type'], how='outer')
    real_data = real_data[real_data['Transfer'] > 0]
    return real_data


def model(params):
    Ub = 10**params.log10Ub
    α = params.α
    β = 10**params.log10β
    fn = evodyn_gamma_v3.simulate(N0=int(1.2e4) , bottleneck=int(GEN_PER_DAY), 
                         generations=550, repeats=100, interval=10,
                         Ub=Ub, tau=1, alpha=α, beta=β, verbose=False)
    df = evodyn_gamma_v3.load(fn, progressbar=False)
    return dict(data=df)


# # Summary statistics
# 
# Based on the *One Biallelic Marker ABC* method from
# > Jorge A., Campos PRAA, Gordo I (2013) An ABC Method for Estimating the Rate and Distribution of Effects of Beneficial Mutations. Genome Biol Evol 5(5):794–806.

def sum_freq_data(freq_data, n_bins=5):
    freq_data = freq_data.copy()
    freq_data = freq_data[freq_data['Transfer'] > 0]
    freq_data['Frequency (abs)'] = abs(0.5 - freq_data['Frequency'])
    pivot = freq_data.pivot_table(index='Population', columns='Transfer', values='Frequency (abs)')
    return np.apply_along_axis(
        lambda x: np.histogram(x, bins=n_bins, range=(0, 0.5))[0]/x.size, 0, pivot.values).T

    
def sum_fit_data(fit_data, max_fitness, n_bins=6):
    fit_data = fit_data[fit_data['Transfer'] > 0]
    pivot = fit_data.pivot_table(index='Population', columns='Transfer', values='Fitness')
    return np.apply_along_axis(
        lambda x: np.histogram(x, bins=n_bins, range=(1, max_fitness))[0]/x.size, 0, pivot.values).T


def distance(real, simulated):
    real, simulated = real['data'], simulated['data']
    max_fitness = real['Fitness'].max()
    real_freq_sum = sum_freq_data(real)
    real_fit_sum = sum_fit_data(real, max_fitness=max_fitness)
    simulated_freq_sum = sum_freq_data(simulated)
    simulated_fit_sum = sum_fit_data(simulated, max_fitness=max_fitness)
    if simulated_fit_sum.shape[0] == 4: # real data only has transfer 20 and 40
        simulated_fit_sum = simulated_fit_sum[[1,3], :]
    freq_resid = real_freq_sum - simulated_freq_sum
    fit_resid = real_fit_sum - simulated_fit_sum
    rss = (freq_resid**2).sum() + (fit_resid**2).sum()
    return rss


# ## Prior distributions
# 
# $$
# log10(U) \sim Uniform[-9, -4] \\
# \alpha \sim Uniform [0.5, 15] \\
# log10(b) \sim Uniform [-4, -0.08]
# $$
# 
# SciPy's uniform expects `loc` and `scale` which are `low` and `width`, rather than `low` and `high`.

if __name__ == '__main__':
    real_data = load_real_data()
    
    prior = Distribution(
        log10Ub=RV("uniform", -9, 5), 
        α=RV("uniform", 0.5, 14.5),
        log10β=RV("uniform", -4, 3.92)
    )

    abc = ABCSMC(
        model, 
        prior,
        distance
    )

    abc_id = abc.new(
        "sqlite:///24hr_dinBminus.db", 
        dict(data=real_data[real_data['type'] == 'dinB-'])
    )
    print('Starting run:', abc_id)
    abc.run(minimum_epsilon=0.7, max_nr_populations=50)
    print('Done.')
