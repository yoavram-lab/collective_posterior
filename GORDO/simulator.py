
import numpy as np
import pandas as pd
from OBMABCSMC import model

import sys
sys.path.append('../src/')
import evodyn_gamma_v3

DILUTION = 1/1000


def model(params, reps=100):
    Ub = 10**params[0]
    α = params[1]
    β = 10**params[2]
    fn = evodyn_gamma_v3.simulate(N0=int(1.2e5) , bottleneck=5, 
                         generations=300, repeats=reps, interval=10,
                         Ub=Ub, tau=1, alpha=α, beta=β, verbose=False)
    df = evodyn_gamma_v3.load(fn, progressbar=False)
    return dict(data=df)


def data_from_simulator(simulation, reps=1):
    y = pd.DataFrame(simulation['data'])
    n = y.shape[0] // reps
    freq_array = y['Frequency'].values.reshape(n, reps)
    fit_array = y['Fitness'].values.reshape(n, reps)
    fit_array = fit_array.clip(min=0, max=2)  # Ensure valid fitness values
    # freq_array = np.abs(0.5-freq_array)  # Use absolute fitness values
    return np.concatenate([freq_array, fit_array]).T


def simulator(params, reps=1):
    x = model(params, reps=reps)
    return data_from_simulator(x, reps=reps)