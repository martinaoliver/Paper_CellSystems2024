#############
###imports#####
#############
import numpy as np
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt


#############
###code#####
#############

np.random.seed(1)

def preLhs(parameterDictList):
    parameterDistributionList = [parameterDistribution(parameterDict,100000) for parameterDict in parameterDictList]
    distributionMinimumLenght = np.amin([len(x) for x in parameterDistributionList])
    croppedParameterDistributionList = [x[:distributionMinimumLenght] for x in parameterDistributionList]
    stackedDistributions = np.column_stack((croppedParameterDistributionList))
    return stackedDistributions

def lhs(data, nsample,seed=1, tqdm_disable=False):
    np.random.seed(seed)
    m, nvar = data.shape
    ran = np.random.uniform(size=(nsample, nvar))
    s = np.zeros((nsample, nvar))
    for j in tqdm(range(0, nvar), disable=tqdm_disable):
        idx = np.random.permutation(nsample) + 1
        P = ((idx - ran[:, j]) / nsample) * 100
        s[:, j] = np.percentile(data[:, j], P)

    if np.any(s<=0):
        print('WARNING: negative values in lhs')
        s[s<0] = 0.001
        
    return s

def loguniform(size, low=-3, high=3):
    return (10) ** (np.random.uniform(low, high, size))



def parameterGaussian( mean, noisetosignal, size):
    stdev = noisetosignal * mean
    gaussianDistribution = np.random.normal(mean, stdev, size)
    return gaussianDistribution

def parameterLogNormal(mean, noisetosignal, size):
    sigma = noisetosignal * mean
    normal_std = np.sqrt(np.log(1 + (sigma/mean)**2))
    normal_mean = np.log(mean) - normal_std**2 / 2
    lognormalDistribution = np.random.lognormal(normal_mean, normal_std, size)
    return lognormalDistribution

def parameterLogUniform( min, max, size):
    loguniformDistribution = loguniform(size)
    croppedLoguniformDistribution = np.array([x for x in loguniformDistribution if min <= x <= max])
    return croppedLoguniformDistribution

def parameterFixed( value, size):
    fixedDistribution = np.full((size), value)
    return fixedDistribution



def parameterDistribution(parameterDict,size):
    if parameterDict['distribution']=='gaussian':
        dist = parameterGaussian(parameterDict['mean'], parameterDict['noisetosignal'],size)
    if parameterDict['distribution']=='lognormal':
        dist = parameterLogNormal(parameterDict['mean'], parameterDict['noisetosignal'],size)
    if parameterDict['distribution']=='loguniform':
        dist =  parameterLogUniform(parameterDict['min'], parameterDict['max'],size)
    if parameterDict['distribution']=='fixed':
        dist =  parameterFixed(parameterDict['value'],size)
    return dist


def plotDist(parameterDictList,lhsDist_df):
    nvar = len(parameterDictList)
  
    fig,axs = plt.subplots(nrows=1,ncols=nvar,figsize=(nvar*5,5))
    for count,parameter in enumerate(parameterDictList):
        name = parameter['name']
        lhsDistColumn = lhsDist_df[name]
        sns.histplot(lhsDistColumn, ax=axs[count], bins=100)
        axs[count].set(ylabel ='',yticks=[],yticklabels=[])
        axs[count].set_xlabel(name, fontsize=15)
    plt.show()


def checkBalance(par_dict, Km_list, KtoV):
    balanceDict = {}
    for Km in Km_list:
        # print(Km)
        Vx =par_dict[KtoV[Km]]
        Kxy = par_dict[Km]
        if Kxy >= 1 and Kxy <= Vx:
            balanceDict[Km] = 'Balanced'
        elif Kxy > 0.1 and Kxy < Vx*10:
            balanceDict[Km] ='Semi balanced'
        elif Kxy <= 0.1 or Kxy >= Vx*10:
            balanceDict[Km] ='Not balanced'
        else:
            print('ERROR!!!!!!!!!')

    if 'Not balanced' in balanceDict.values():
        return 'Not balanced'
    elif 'Semi balanced'  in balanceDict.values():
        return 'Semi balanced'
    elif all(x == 'Balanced' for x in balanceDict.values()):
        return 'Balanced'
    