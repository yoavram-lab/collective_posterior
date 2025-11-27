# collective_posterior

Repository of the paper: "Inferring a Collective Posterior Distribution from Highly Variable Experimental Replicates"

The collective posterior is an abstract extension to simulation-based inference to allow inference from multiple observation with any density estimator for indiviudal posterior distribution estimation.

The collective posterior implementation itself is in "collective_posterior.py".

To reproduce the paper's benchmark results, you need to install sbibm and sbi, by running the following commands in the following order in your terminal:

```pip install sbibm```
```pip install sbi==0.23.1```
```pip install torch==1.13.1```

To reproduce the WF (Chuong et al., 2025) or EVO_SIM (3-locus benchmark) results, you don't need to install sbibm. Simply install sbi and torch and run tests/inference.

Each simulator has its own folder:
- The notebook test_collective.ipynb includes the assessment figures and an example of inference from synthetic data.

For WF:
- The notebooks wf_collective_abc.ipynb and empirical_wf.ipynb demonstrate inference from empirical data with Rej-ABC and NPE, respectively. 
- The notebook erratic.ipynb demonstrates the "randomly low posterior densities" for unlikely parameters.
- The notebook npse.ipynb compares NPE and NPSE on empirical data.
- The notebook empirical_wf.ipynb shows the Collective Posterior vs. NPE+PIE vs. NPSE on empirical data.

Additionally:
- The notebook collective_sparrows.ipynb shows the collective inference for Lachlan et al., 2018's data.
- The notebook phylo_abc.ipynb demonstrates inference with ABC and the collective posterior on a phylogenetic dataset.

Inference files:
- inference.py - NPE (individual obs)
- inference_sbi_iid.py - NPE+PIE
- inference_utils.py - priors
