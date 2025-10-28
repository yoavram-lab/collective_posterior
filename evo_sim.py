import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, List
import torch

@dataclass
class SimConfig:
    N: int = 10_000                      # population size (haploid)
    s: Tuple[float, float, float] = (0.02, 0.01, 0.03)  # selection coefficients per mutation
    u: Tuple[float, float, float] = (1e-6, 5e-6, 1e-5)  # forward mutation rates (0->1) per locus
    gens: int = 1000                     # total generations
    sample_every: int = 100              # observe every k generations (including gen 0)
    seed: int = 12345                    # RNG seed
    allow_back_mutation: bool = False    # keep False as requested; can switch to True if needed
    v_back: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # back mutation (1->0), ignored if allow_back_mutation=False

def simulate_three_mutations(cfg: SimConfig) -> Dict[str, np.ndarray]:
    """
    Wright–Fisher haploid simulation with 3 binary loci (0/1).
    Genotypes are indexed 0..7 by their 3-bit representation:
      bit0 -> locus A (mutation 1), bit1 -> locus B (mutation 2), bit2 -> locus C (mutation 3).
    Fitness is multiplicative: w(g) = Π_i (1 + s_i)^{bit_i}.
    Mutation is applied independently per locus during reproduction (only forward 0->1 by default).
    Drift is modeled via multinomial sampling.
    Returns a dict with times and allele frequencies for each locus at observation points.
    """

    rng = np.random.default_rng(cfg.seed)

    # All 8 genotypes and their bit representation
    genotypes = np.array([[(g >> 0) & 1, (g >> 1) & 1, (g >> 2) & 1] for g in range(8)], dtype=np.int8)  # [8,3]

    # Fitness per genotype (multiplicative)
    s = np.asarray(cfg.s, dtype=np.float64)  # [3]
    w = np.prod((1.0 + s) ** genotypes, axis=1)  # [8]

    # Start with all wildtype (000)
    counts = np.zeros(8, dtype=np.int64)
    counts[0] = cfg.N

    # Observation schedule
    times = np.arange(0, cfg.gens + 1, cfg.sample_every)
    rec_A, rec_B, rec_C = [], [], []

    def record():
        # allele frequency for each locus = sum over genotypes of bit * freq
        freqs = counts / counts.sum()
        pA = (genotypes[:, 0] * freqs).sum()
        pB = (genotypes[:, 1] * freqs).sum()
        pC = (genotypes[:, 2] * freqs).sum()
        rec_A.append(pA); rec_B.append(pB); rec_C.append(pC)

    record()  # gen 0

    # Precompute mutation probabilities per parent genotype -> child genotype
    # For each parent genotype g (bits b0,b1,b2), the child genotype is formed by
    # flipping each bit independently with prob u_i if bit==0 (and optionally v_i if bit==1).
    # Build an 8x8 transition matrix M where M[g, h] = P(child=h | parent=g, mutation only)
    u = np.asarray(cfg.u, dtype=np.float64)
    if cfg.allow_back_mutation:
        v = np.asarray(cfg.v_back, dtype=np.float64)
    else:
        v = np.zeros(3, dtype=np.float64)

    M = np.zeros((8, 8), dtype=np.float64)
    for g in range(8):
        b = genotypes[g]  # [3]
        # Child genotype h bits are independent conditioned on parent bits
        # P(bit_i -> 1) = (1 - v_i) if b_i=1 else u_i
        p1 = np.where(b == 1, 1.0 - v, u)  # prob child bit=1
        p0 = 1.0 - p1
        # Enumerate h and compute product over loci
        for h in range(8):
            hb = genotypes[h]
            probs = np.where(hb == 1, p1, p0)
            M[g, h] = probs.prod()

        # numerical safety
        M[g, :] /= M[g, :].sum()

    for t in range(1, cfg.gens + 1):
        N = counts.sum()
        if N != cfg.N:
            # keep size exactly N via multinomial (Wright-Fisher)
            # (N should always equal cfg.N though)
            pass

        # Selection: sample parents with probability ∝ counts * w
        weights = counts * w  # unnormalized
        if weights.sum() == 0:
            # if extinct (shouldn't happen), reset to wildtype
            counts[:] = 0
            counts[0] = cfg.N
            weights = counts * w

        p_sel = weights / weights.sum()

        # Draw number of offspring from each parent genotype
        offspring_from_parent = rng.multinomial(cfg.N, p_sel)  # [8]

        # Mutation: each parent g produces children distributed across h via M[g,:]
        # Accumulate expected children by multionomial draws per parent
        new_counts = np.zeros(8, dtype=np.int64)
        for g in range(8):
            k = offspring_from_parent[g]
            if k > 0:
                new_counts += rng.multinomial(k, M[g])

        counts = new_counts

        # Record if on schedule
        if (t % cfg.sample_every) == 0:
            record()

    return {
        "times": times,                       # shape [T]
        "p_mut1": np.array(rec_A),            # shape [T]
        "p_mut2": np.array(rec_B),            # shape [T]
        "p_mut3": np.array(rec_C),            # shape [T]
    }


def simulate_three_mutations_vec30(cfg: SimConfig) -> np.ndarray:
    out = simulate_three_mutations(cfg)
    # take generations 100..1000 (skip gen 0), that’s 10 points
    f1 = out["p_mut1"][1:]  # shape (10,)
    f2 = out["p_mut2"][1:]  # shape (10,)
    f3 = out["p_mut3"][1:]  # shape (10,)
    vec30 = np.concatenate([f1, f2, f3], axis=0).astype(np.float64)  # shape (30,)
    return vec30

def evo_sim(theta):
    s1, s2, s3, u1, u2, u3 = 10**theta
    cfg = SimConfig(
        N=100_000,
        s=(s1, s2, s3),
        u=(u1, u2, u3),
        gens=1000,
        sample_every=100,
        seed=None
    )
    v = simulate_three_mutations_vec30(cfg)
    return v


def EVO_SIM_wrapper(reps, parameters, seed=None):
    evo_reps = torch.empty(reps, 30)
    for i in range(reps):
        out=evo_sim(parameters)
        evo_reps[i,:] = torch.tensor(out)
    return evo_reps