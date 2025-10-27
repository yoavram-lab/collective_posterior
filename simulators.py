import numpy as np
import torch
import sbibm


# constants
N = int(1e7)
generation = np.array([8, 21, 29, 37, 50, 58, 66, 79, 87, 95, 108, 116]) # from Chuong et al 2024
        


def CLASSIC_WF(parameters, seed=None, generation=torch.arange(0,201,10, dtype=int)):
    """ Classic WF evolution simulator
    Simulates evolutionary dynamics for x generations
    Returns proportion of the population with a mutation for specific generations
    
    Parameters
    -------------------
    N : int = 
        population size  
    s : float
        fitness benefit of mutations  
    m : float 
        probability mutation to mutation   
    generation : np.array, 1d 
        with generations to output
    seed : int
    """
    # SNV parameters
    s, m, N = 10**parameters.numpy()
    N = int(N)
    
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    # Order is: wt, snv
    
    w = np.array([1, 1 + s], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m, 0],
                [m, 1]], dtype='float64')
    assert np.allclose(M.sum(axis=0), 1)
    
    
    # mutation and selection
    E = M @ S

    # rows are genotypes, p has proportions after initial (unreported) growth
    n = np.zeros(2)
    n[0] = N # wt

    # follow proportion of the population with mutation
    # here rows will be generation, columns (there is only one) is replicate population
    p_mut = []
    
    # run simulation to generation _max
    for t in range(int(generation.max()+1)):    
        p = n/N  # counts to frequencies
        p_mut.append(p[1])  # frequency of reported mutations
        p = E @ p.reshape((2, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    ret = np.transpose(p_mut)[generation.numpy().astype(int)]
    return torch.tensor(ret)


def WF(parameters, seed=None):
    """ CNV evolution simulator
    Simulates CNV and SNV evolution for x generations
    Returns proportion of the population with a CNV for specific generations
    
    Parameters
    -------------------
    N : int = 10M
        population size  
    s_snv : float
        fitness benefit of SNVs  
    m_snv : float 
        probability mutation to SNV   
    generation : np.array, 1d 
        with generations to output
    seed : int
    
    s_cnv : float
        fitness benefit of CNVs  
    m_cnv : float 
        probability mutation to CNV 
    p_0: float
        fraction of population with GAP1 CNV before beginning of experiment
    """
    # CNV parameters
    s_cnv, m_cnv, p_0 = 10**parameters

    # SNV parameters as constants
    s_snv = 1e-3
    m_snv = 1e-5
    
    if seed is not None:
        np.random.seed(seed=seed)
    else:
        np.random.seed()

    
    
    
    # Order is: wt, cnv+, cnv-, snv
    
    w = np.array([1, 1 + s_cnv, 1 + s_cnv, 1 + s_snv], dtype='float64')
    S = np.diag(w)
    
    # make transition rate array
    M = np.array([[1 - m_cnv - m_snv, 0, 0, 0],
                [m_cnv, 1, 0, 0],
                [0, 0, 1, 0],
                [m_snv, 0, 0, 1]], dtype='float64')
    assert np.allclose(M.sum(axis=0), 1)
    
    
    # mutation and selection
    E = M @ S

    # rows are genotypes, p has proportions after initial (unreported) growth
    n = np.zeros(4)
    n[2] = N*p_0 # cnv-
    n[0] = N*(1-p_0) # wt
    
    # follow proportion of the population with CNV
    # here rows will be generation, columns (there is only one) is replicate population
    p_cnv = []
    
    # run simulation to generation 116
    for t in range(int(generation.max()+1)):    
        p = n/N  # counts to frequencies
        p_cnv.append(p[1])  # frequency of reported CNVs
        p = E @ p.reshape((4, 1))  # natural selection + mutation        
        p /= p.sum()  # rescale proportions
        n = np.random.multinomial(N, np.ndarray.flatten(p)) # random genetic drift
    ret = np.transpose(p_cnv)[generation.astype(int)]
    noise = np.random.normal(0,0.02,size=ret.shape)
    return torch.tensor(ret)

def wrapper(simulator, reps, parameters, seed=None):
    rep_1 = simulator(parameters)
    if simulator in [WF, CLASSIC_WF]:
        rep_1 = rep_1.reshape(1,-1)
    out_reps = torch.empty((reps, rep_1.shape[1]))
    out_reps[0,:] = rep_1
    for i in range(1,reps):
        out=simulator(parameters)
        out_reps[i,:] = out
    return out_reps

def wrapper_hierarchical(simulator, reps, parameters, var=0.02, seed=None):
    rep_1 = simulator(parameters)
    if simulator in [WF, CLASSIC_WF]:
        rep_1 = rep_1.reshape(1,-1)
    out_reps = torch.empty((reps, rep_1.shape[1]))
    out_reps[0,:] = rep_1
    for i in range(1,reps):
        out=simulator(parameters + torch.normal(0,var,size=parameters.shape))
        out_reps[i,:] = out
    return out_reps

def WF_wrapper(reps, parameters, seed=None):
    evo_reps = torch.empty(reps, len(generation))
    for i in range(reps):
        out=WF(parameters, seed=seed)
        evo_reps[i,:] = out
    return evo_reps

=======
def FWDPY_wrapper(reps, parameters):
    evo_reps = torch.empty(reps, 20)
    for i in range(reps):
        out=FWDPY(parameters)
        evo_reps[i,:] = out
    return evo_reps
>>>>>>> 26bf04cad3ed182f5a1997ba28c7427eadfea492

def CLASSIC_WF_wrapper(reps, parameters, seed=None, generation=torch.arange(0,201,10)):
    evo_reps = torch.empty(reps, len(generation))
    for i in range(reps):
        out= CLASSIC_WF(parameters, seed=seed)
        evo_reps[i,:] = out
    return evo_reps

glu_task = sbibm.get_task('gaussian_linear_uniform')
glu_simulator = glu_task.get_simulator()

def GLU(parameters):
    return glu_simulator(parameters)

def GLU_wrapper(reps, parameters):
    glu_reps = torch.empty(reps, 10)
    for i in range(reps):
        out=GLU(parameters)
        glu_reps[i,:] = out
    return glu_reps


### SLCP ###
slcp_task = sbibm.get_task('slcp') # See sbibm.get_available_tasks() for all tasks

slcp_simulator = slcp_task.get_simulator()

def SLCP(parameters):
    return slcp_simulator(parameters)

def SLCP_wrapper(reps, parameters):
    slcp_reps = torch.empty(reps, 8)
    for i in range(reps):
        out=SLCP(parameters)[0]
        slcp_reps[i,:] = out
    return slcp_reps

# def GORDO_wrapper(reps, parameters):
#     gordo_reps = torch.empty(reps, 11)
#     for i in range(reps):
#         out=GORDO(parameters.numpy())
#         gordo_reps[i,:] = torch.tensor(out)
#     return gordo_reps
