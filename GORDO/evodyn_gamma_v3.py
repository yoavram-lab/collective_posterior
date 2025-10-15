import os
import subprocess
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import pandas as pd
import numpy as np
from numpy.random import uniform, randint

import click
from tqdm import tqdm

folder = os.path.dirname(__file__)
if not folder:
	folder = './'
source_code = os.path.join(folder, 'evodyn_gamma_v3.cpp')
executable = os.path.join(folder, 'evodyn_gamma')


def run_subprocess(cmd, verbose=True):
	if verbose: print("Running command: {}".format(cmd))
	cmd = cmd.split()
	try:
		proc = subprocess.run(cmd, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
	except Exception as e:
		raise click.ClickException("Failed: {}".format(e))
	else:
		if proc.returncode != 0:
			raise click.ClickException(
				"Failed with return code {}:\n{}\n{}".format(
					proc.returncode, proc.stdout.decode(), proc.stderr.decode()))
		if verbose:
			print("Succeeded")
			if proc.stdout:
				print(proc.stdout.decode())
	return proc


@click.group()
def main():
	pass

@main.command()
def path():
	print(source_code)
	print(executable)


@main.command()
def compile():
	# on mac, install gsl with: brew install gsl
	# on ubuntu, install gsl with: apt-get install libgsl-dev
	cmd = f'c++ {source_code} -o {executable} -lm -lgsl -lgslcblas -I/Users/yoavram/homebrew/include -L/Users/yoavram/homebrew/lib'
	run_subprocess(cmd)


@main.command()
@click.option('--n0', type=int, default=100000)
@click.option('--bottleneck', type=int, default=5)
@click.option('--generations', type=int, default=300)
@click.option('--repeats', type=int, default=100)
@click.option('--interval', type=int, default=10)
@click.option('--Ub', type=float, default=3e-5)
@click.option('--tau', type=float, default=1)
@click.option('--alpha', type=float, default=1)
@click.option('--beta', type=float, default=0.01)
@click.option('--seed', type=int, default=1)
def run(n0, bottleneck, generations, repeats, interval, ub, tau, alpha, beta, seed):
	N0 = n0
	Ub = ub
	simulate(N0, bottleneck, generations, repeats, interval, Ub, tau, alpha, beta, seed, verbose=True)


@main.command()
@click.option('--sets', type=int, default=100)
def explore(sets):
    Ubs = 10**uniform(-9, -4, size=sets)
# 	taus = 10**uniform(0, 2, size=sets)
    alphas = uniform(0.5, 15, size=sets)
    betas = 10**uniform(-4, -0.08, size=sets)
    N0 = int(1.2e4)
    DILUTION = 1/10000
    GEN_PER_DAY = np.log2(1/DILUTION)
    bottleneck = int(GEN_PER_DAY)
    generations = 550
    repeats = 100
    interval = 10
    tau = 1
    seeds = range(sets)
    def _simulate(Ub, alpha, beta, seed):
        return simulate(N0, bottleneck, generations, repeats, interval, Ub, tau, alpha, beta, seed, verbose=False)
    with ThreadPoolExecutor() as exec:
        exec_map = exec.map(_simulate, Ubs, alphas, betas, seeds)
        for _ in tqdm(exec_map, 'Simulation sets', total=sets):
            pass


def simulate(N0, bottleneck, generations, repeats, interval, Ub, tau, alpha, beta, seed=None, verbose=False):
	Nmax = N0 * 2**bottleneck
	if not os.path.exists('output'):
		os.mkdir('output')
	if seed is None:
		seed = randint(0, 2**32-1)
    # args: <N0> <NMax> <BottleneckGenerations> <Log Interval> <Max Generations> <Simulation Repeats> <Ub> <tau> <alpha> <beta> <random_seed_gsl>
	cmd = f'{executable} {N0:d} {Nmax:d} {bottleneck:d} {interval:d} {generations:d} {repeats:d} {Ub:g} {tau:g} {alpha:g} {beta:g} {seed:d}'
	proc = run_subprocess(cmd, verbose)
	return proc.stdout.decode().split()[-1] # return output filename


@main.command()
@click.option('--pattern', type=str)
@click.option('--output', type=click.Path(file_okay=True, writable=True))
@click.option('--bottleneck', type=int, default=5)
def collate(pattern, output, bottleneck):
    print("Loading files: {}".format(pattern))
    df = load(pattern, bottleneck=bottleneck)
    print("Writing collated data to {}".format(output))
    # if output.endswith('gz'):
    #     df.to_csv(output, compression='gzip', index=False)
    # else:
    #     df.to_csv(output, index=False)


def load(pattern=os.path.join('output', '*.txt'), progressbar=True):
    filenames = glob(pattern)
    if progressbar:
    	filenames = tqdm(filenames, 'Files')
    long_df = None
    for fname in filenames:
        df = pd.read_table(fname, sep='\t')
        df.rename(columns={'Population ': 'Population'}, inplace=True)
        df = pd.melt(df, id_vars='Population', var_name='Measurement', value_name='Value')
        df['Transfer'] = [int(x.split()[-1]) for x in df['Measurement']]
        df['Measurement'] = [str.join(' ', x.split()[:-1]) for x in df['Measurement']]
        fdf = df[df['Measurement'] == 'Frequency at Bottleneck'].rename(columns={'Value':'Frequency'}).drop(labels='Measurement', axis=1)
        wdf = df[df['Measurement'] == 'Whole Population Fitness Average at Bottleneck'].rename(columns={'Value':'Fitness'}).drop(labels='Measurement', axis=1)
        df = pd.merge(fdf, wdf, on=('Population', 'Transfer'))
        df['Filename'] = fname
        basename = os.path.splitext(os.path.split(fname)[-1])[0]
        # SummaryStatistics_Ub_%g_tau_%g_alpha_%g_beta_%g
        # popul.Ub, popul.tau, popul.alpha, popul.beta
        _,_,Ub,_,tau,_,alpha,_,beta = basename.split('_')
        df['Ub'] = float(Ub)
        df['τ'] = float(tau)
        df['α'] = float(alpha)
        df['β'] = float(beta)
        long_df = pd.concat((long_df, df)) if long_df is not None else df
    return long_df


if __name__ == '__main__':
	main()

