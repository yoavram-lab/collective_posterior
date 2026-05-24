# collective_posterior

Code and notebooks for the paper **"Collective Posterior Inference from Highly Variable Empirical Replicates"**.

The collective posterior extends simulation-based inference from a single observation to a set of replicate observations. The core implementation is in `collective_posterior.py`; it wraps an amortized individual posterior estimator and combines replicate-wise evidence into a collective posterior.

## Repository Layout

- `collective_posterior.py`: collective posterior implementation and samplers.
- `simulators.py`: WF, GLU, and SLCP simulators and wrappers.
- `evo_sim.py`: 3-locus evolutionary simulator and wrappers.
- `inference.py`: trains an individual-observation NPE posterior.
- `inference_sbi_iid.py`: trains NPE with a permutation-invariant embedding for replicate sets, referred to in the notebooks as NPE+PIE.
- `test_posterior.py`: evaluates synthetic benchmarks and saves accuracy, coverage, and posterior samples.
- `installation_check.ipynb`: smoke test for the installed environment.
- `GLU/`, `SLCP/`, `EVO_SIM/`, `WF/`: benchmark-specific notebooks, posteriors, cached samples, and figures.

## Environment

The paper notebooks are tested with **Python 3.10** and the pinned packages in `requirements.txt`.

Important pinned dependencies:

- `torch==2.5.1+cpu`
- `sbi==0.25.0`
- `scikit-learn==1.5.0`
- `seaborn==0.13.2`

The CPU PyTorch wheel is intentional: it avoids CUDA/cuDNN runtime mismatches in notebook kernels. `sbi==0.25.0` is used because the NPSE workflow needs recent `sbi` functionality, including NPSE/iid support. Use the same environment for loading the saved `.pkl` posterior files and for running the notebooks.

The local GLU and SLCP implementations follow the corresponding `sbibm` task definitions, so `sbibm` is not required at runtime. `phylo_abc.ipynb` is outside the pinned paper environment and still requires the external `GenomeRearrangement` package separately.

## Installation With Conda

From this directory:

```bash
conda create -n collective python=3.10 -y
conda activate collective
python -m pip install --upgrade pip
python -m pip install  -r requirements.txt
python -m pip install sbibm==1.1.0 --no-deps
python -m ipykernel install --user --name collective --display-name "Python (collective)"```


## Installation Check

After installing, open and run:

```bash
jupyter lab installation_check.ipynb
```

Run `installation_check.ipynb` from the repository root. It checks package imports and versions, simulator outputs, posterior loading, and a small collective-posterior sampling call. If the saved notebook output shows old package versions, re-run all cells with the new kernel.

## Running Notebooks

Many notebooks use relative paths such as `posteriors/...`, `tests/...`, or `x_test.pt`. Run benchmark notebooks from their own directory.

The repository includes cached `.pkl`, `.pt`, `.csv`, and figure files for most results. Use these cached files for fast reproduction of the reported plots. Retraining or resampling can produce slightly different numerical values unless you also reproduce the original random seeds and sampling settings.

## Reproducing GLU and SLCP Results

Fast path with cached results:

```bash
cd GLU
jupyter lab "test collective.ipynb"

cd ../SLCP
jupyter lab "test collective.ipynb"
```

These notebooks reproduce the synthetic GLU and SLCP figures from saved posterior and test files:

- `GLU/posteriors/posterior_GLU_100000_20.pkl`
- `GLU/posteriors/posterior_iid_GLU_100000_20.pkl`
- `GLU/tests/test_theta.pt`
- `GLU/tests/test_x.pt`
- `GLU/tests/accus_*.pt`, `GLU/tests/covs_*.pt`, `GLU/tests/samples_*.pt`
- corresponding files under `SLCP/posteriors/` and `SLCP/tests/`

To regenerate the trained posteriors from the repository root:

```bash
python inference.py -m GLU -n 100000 -e 20
python inference_sbi_iid.py -m GLU -n 100000 -e 20

python inference.py -m SLCP -n 100000 -e 20
python inference_sbi_iid.py -m SLCP -n 100000 -e 20
```

The training scripts save new posterior files under the model folder with their built-in filenames. Move or rename them into `posteriors/` if you want the notebooks to use regenerated posteriors without editing notebook paths.

To regenerate synthetic evaluation tensors, use `test_posterior.py`. Pass `-cp` for the Collective Posterior condition and omit it for NPE+PIE. For example:

```bash
python test_posterior.py -m SLCP -p SLCP/posteriors/posterior_SLCP_100000_20.pkl -s 1000 -t SLCP/tests/test_theta.pt -x SLCP/tests/test_x.pt -cp -e _adaptive
python test_posterior.py -m SLCP -p SLCP/posteriors/posterior_iid_SLCP_100000_20.pkl -s 1000 -t SLCP/tests/test_theta.pt -x SLCP/tests/test_x.pt
```

The script writes new `accus_*`, `covs_*`, and `samples_*` files under the model folder. Move them into the corresponding `tests/` folder or update the notebook paths if you want the notebooks to use regenerated files.

## Reproducing EVO_SIM Results

Fast path with cached results:

```bash
cd EVO_SIM
jupyter lab "test collective.ipynb"
jupyter lab epsilon_vs_N.ipynb
```

Main cached files:

- `posterior_EVO_SIM_30000_20.pkl`
- `posterior_iid.pkl`
- `theta_test.pt`
- `x_test.pt`, `x_test_h.pt`, `x_test_n.pt`, `x_test_r.pt`
- `accus_EVO_SIM_*.pt`, `covs_EVO_SIM_*.pt`, `samples_EVO_SIM_*.pt`
- `accus_npse*.pt`, `covs_npse*.pt`, `samples_npse*.pt`
- `epsilon_vs_N_results_full.pt`
- `epsilon_testset_estimates_full.pt`

Notebook roles:

- `simulate.ipynb`: generates the EVO_SIM synthetic test tensors.
- `test collective.ipynb`: reproduces the main synthetic EVO_SIM comparison between Collective Posterior, NPE+PIE, and NPSE.
- `epsilon_vs_N.ipynb`: reproduces the epsilon-by-number-of-replicates sweep. The sweep caches each `(epsilon, N)` result to `epsilon_vs_N_results_full.pt`, so interrupted runs can be resumed.
- `epsilon_fitting.ipynb`: explores epsilon estimation.

To regenerate the individual NPE and NPE+PIE posteriors from the repository root:

```bash
python inference.py -m EVO_SIM -n 30000 -e 20
python inference_sbi_iid.py -m EVO_SIM -n 30000 -e 20
```

The training scripts save new posterior files under `EVO_SIM/` with their built-in filenames. Rename them if you want to replace the cached posterior files loaded by the notebooks.

To regenerate Collective Posterior and NPE+PIE evaluation samples:

```bash
python test_posterior.py -m EVO_SIM -p EVO_SIM/posterior_EVO_SIM_30000_20.pkl -s 1000 -t EVO_SIM/theta_test.pt -x EVO_SIM/x_test_r.pt -cp -e _adaptive
python test_posterior.py -m EVO_SIM -p EVO_SIM/posterior_iid.pkl -s 1000 -t EVO_SIM/theta_test.pt -x EVO_SIM/x_test_r.pt
```

The NPSE comparison files are cached in `EVO_SIM/`. If you retrain NPSE, use `EVO_SIM/inference_npse.py` as the starting point and save the regenerated `accus_npse`, `covs_npse`, and `samples_npse` files expected by `test collective.ipynb`.

## Reproducing WF Synthetic Results

Fast path with cached results:

```bash
cd WF
jupyter lab "test collective.ipynb"
```

Main cached files:

- `WF/posteriors/posterior_WF_30000_20.pkl`
- `WF/posteriors/posterior_iid_WF_30000_20.pkl`
- `WF/tests/theta_test.pt`
- `WF/tests/x_test.pt`
- `WF/accus_WF_adaptive.pt`, `WF/covs_WF_adaptive.pt`, `WF/samples_WF_adaptive.pt`
- `WF/accus_WF_iid.pt`, `WF/covs_WF_iid.pt`, `WF/samples_WF_iid.pt`

To regenerate the posteriors from the repository root:

```bash
python inference.py -m WF -n 30000 -e 20
python inference_sbi_iid.py -m WF -n 30000 -e 20
```

The training scripts save new posterior files under `WF/` with their built-in filenames. Move or rename them into `WF/posteriors/` if you want to replace the cached posterior files loaded by the notebooks.

To regenerate the synthetic benchmark samples and coverage files:

```bash
python test_posterior.py -m WF -p WF/posteriors/posterior_WF_30000_20.pkl -s 1000 -t WF/tests/theta_test.pt -x WF/tests/x_test.pt -cp -e _adaptive
python test_posterior.py -m WF -p WF/posteriors/posterior_iid_WF_30000_20.pkl -s 1000 -t WF/tests/theta_test.pt -x WF/tests/x_test.pt
```

## Reproducing WF Empirical Results

WF empirical data are in `WF/empirical_data/`.

Recommended order:

```bash
cd WF
jupyter lab npse.ipynb
jupyter lab empirical_wf.ipynb
```

`npse.ipynb` trains or loads NPSE, evaluates it on the empirical WF datasets, runs the NPSE inference-cycle check, and saves posterior samples:

- `posteriors/posterior_npse.pkl`
- `tests/samples_npse_<dataset>.pt`
- `tests/cycle_samples_npse_<dataset>.pt`
- `tests/cycle_observations_npse_<dataset>.pt`
- `tests/npse_empirical_summary.csv`
- `tests/npse_inference_cycle_summary.csv`

Set `FORCE_TRAIN_NPSE = False` in `npse.ipynb` if you want to load the cached `posteriors/posterior_npse.pkl` instead of retraining.

`empirical_wf.ipynb` compares Collective Posterior, NPE+PIE, and NPSE on the empirical WF datasets. By default it reuses saved posterior samples:

- `REUSE_SAVED_SAMPLES = True`
- `FORCE_RECOMPUTE_SAMPLES = False`

The notebook loads or saves:

- `tests/samples_collective_<dataset>.pt`
- `tests/samples_npe_pie_<dataset>.pt`
- `tests/samples_npse_<dataset>.pt`
- `tests/cycle_samples_collective_<dataset>.pt`
- `tests/cycle_samples_npe_pie_<dataset>.pt`
- `tests/cycle_samples_npse_<dataset>.pt`
- `tests/cycle_ovl_ks.csv`
- `tests/posterior_predictive_interval_coverage_200.csv`
- `tests/posterior_predictive_interval_coverage_mae_200.csv`
- `tests/predictive_checks_summary.png`
- `tests/predictive_checks_summary.tif`
- `tests/predictive_checks_summary.pdf`

`wf_collective_abc.ipynb` contains the Rej-ABC empirical WF analysis. `epsilon_fitting.ipynb` and `erratic.ipynb` provide WF-specific epsilon and posterior-density diagnostics.

## License

This repository is licensed under the MIT License. See `LICENSE`.
