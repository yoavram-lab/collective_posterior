import numpy as np
import pandas as pd


def WF(s_cnv, m_cnv, p0, generation, seed=None, N=1.9e6, noisy=False):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for x generations
    Returns proportion of the population with a CNV for specific generations
    
    Parameters
    -------------------
    N : int
        population size  
        
    generation : np.array, 1d 
        with generations to output
        
    seed : int
    
    s_cnv : float
        fitness benefit of non-CNVs  
    m_cnv : float 
        CNV reversion rate
    p0: initial CNV proportion
    """
    s_cnv, m_cnv, p0 = 10**np.array([s_cnv, m_cnv, p0])
    
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert N > 0
    N = np.uint64(N)    
    
    # Order is: cnv, non-cnv
    
    w = np.array([1, 1 + s_cnv], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m_cnv, 0],
                  [m_cnv, 1]], dtype='float64')
    assert np.allclose(M.sum(axis=0), 1)
    
    
    # mutation and selection
    E = M @ S

    # rows are genotypes, p has proportions after initial (unreported) growth
    n = np.zeros(2)
    n[0] = N*(1-p0) # cnv
    n[1] = N*p0 # cnv reversions
    
    # follow proportion of the population with CNV
    # here rows will be generation, columns (there is only one) is replicate population
    p_cnv = []
    
    # run simulation to generation G
    for t in range(int(generation.max()+1)):    
        p = n/N  # counts to frequencies
        p_cnv.append(p[0])  # frequency of reported CNVs
        p = E @ p.reshape((2, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    
    res = np.transpose(p_cnv)[generation.astype(int)]
    
    # Gaussian noise
    if noisy:
        res = res + np.random.normal(loc=0, scale=0.02, size=(len(generation),))
    return res



def combined_WF(s_cherry, m_cherry, p0_cherry, s_citrine, m_citrine, p0_citrine, s_cc, p0_cc, generation, seed=None, N=1.9e6, noisy=False):
    """ CNV evolution simulator
    Simulates epistatic CNV evolution for x generations
    Returns proportion of the population with a CNV for specific generations of both CNV types
    
    Parameters
    -------------------
    N : int
        population size  
        
    generation : np.array, 1d 
        with generations to output
        
    seed : int
    
    s_ : float
        fitness benefit of non-CNVs  
    m_ : float 
        CNV reversion rate
    p0_: initial CNV proportion
    """

    s_cherry, m_cherry, p0_cherry, s_citrine, m_citrine, p0_citrine, s_cc, p0_cc = 10**np.array([s_cherry, m_cherry, p0_cherry, s_citrine, m_citrine, p0_citrine, s_cc, p0_cc])
        
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    assert N > 0
    N = np.uint64(N)    
    
    # Order is: cnv, non-cnv
    
    w = np.array([1, 1 + s_cherry, 1 + s_citrine, 1 + s_cc], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m_cherry - m_citrine, 0, 0, 0], # cherry V citrine V
                  [m_cherry, 1-m_citrine, 0, 0], # cherry X citrine V
                 [m_citrine, 0, 1-m_cherry, 0], # cherry V citrine X
                 [0, m_citrine, m_cherry, 1]], dtype='float64') # cherry X citrine X
    assert np.allclose(M.sum(axis=0), 1)
    
    # mutation and selection
    E = M @ S

    # rows are genotypes, p has proportions after initial (unreported) growth
    n = np.zeros(4)
    n[1] = N*p0_cherry 
    n[2] = N*p0_citrine 
    n[3] = N*p0_cc 
    n[0] = N-n[1]-n[2]-n[3] 
              
    # follow proportion of the population with CNV
    # here rows will be generation, columns (there is only one) is replicate population
    p_all = []
    # run simulation to generation G
    for t in range(int(generation.max()+1)):    
        p = n/N  # counts to frequencies
        p_all.append(p)  # frequency of reported CNVs
        p = E @ p.reshape((4, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    
    res = np.array(p_all)[generation.astype(int),:]
    total_citrine = (res[:,1] + res[:,0]) # citrine present = missing cherry + none missing
    total_cherry = (res[:,2] + res[:,0]) # missing citrine + none missing
    ret = np.concatenate([total_cherry, total_citrine])
    
    # Gaussian noise
    if noisy:
        ret = ret + np.random.normal(loc=0, scale=0.02, size=(2*len(generation),))
    return ret
