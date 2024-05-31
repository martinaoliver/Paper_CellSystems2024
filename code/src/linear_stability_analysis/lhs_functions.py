import numpy as np
#This function return a loguniform distribution with values from 10^low to 10^high
def loguniform(low=-3, high=3, size=None):
    return (10)**(np.random.uniform(low, high, size))

# - This function accepts an already existing distribution and a required number of samples
# and outputs an array with samples distributed in a latin-hyper-cube sampling manner.

def lhs_list(data,nsample,seed=False):
    if seed != False:
        np.random.seed(seed)

    m,nvar = data.shape
    ran=np.random.uniform(size=(nsample,nvar))
    s=np.zeros((nsample,nvar))
    for j in range(0,nvar):
        idx=np.random.permutation(nsample)+1
        P=((idx-ran[:,j])/nsample)*100
        s[:,j]= np.percentile(data[:,j],P)
    return s

# - Input number of initial conditions needed and the number of species in each sample obtain an array with the
# initial conditions distributed in a lhs manner.
def lhs_initial_conditions(n_initialconditions,n_species):
    data = np.column_stack(([loguniform(size=100000)]*n_species))
    initial_conditions = lhs_list(data,n_initialconditions, seed=1)
    return initial_conditions
