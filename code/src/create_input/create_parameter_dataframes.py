#############
### own functions ###
#############

from parameter_creator_functions import *


#############
### imported packages###
#############

import pandas as pd
import pickle as pkl


#Name circuit and variant to idenfiy the parameter sets
circuit_n=1
variant='1'


#Define the parameter ranges and values
minV = 1;maxV=100;
minb=0.1;maxb=1
minK=0.1;maxK=250
K1=0.0183; K2=0.0183
DUmin=0.1; DUmax=10; DVmin=0.1; DVmax=10
muU=0.0225; muV=0.0225
KdiffpromMin=0.1;KdiffpromMax=250
muLVA_estimate =1.143
muAAV_estimate =0.633
muASV_estimate=0.300 #this corresponds to mua

#Maximum production parameters (V)
Va = {'name':'Va','distribution':'loguniform', 'min':minV/maxb, 'max':maxV/minb}
Vb = {'name':'Vb','distribution':'loguniform', 'min':minV/maxb, 'max':maxV/minb}
Vc = {'name':'Vc','distribution':'loguniform', 'min':minV/maxb, 'max':maxV/minb}
Vd = {'name':'Vd','distribution':'loguniform', 'min':minV/maxb, 'max':maxV/minb}
Ve = {'name':'Ve','distribution':'loguniform', 'min':minV/maxb, 'max':maxV/minb}
Vf = {'name':'Vf','distribution':'loguniform', 'min':minV/maxb, 'max':maxV/minb}
V_parameters = [Va, Vb, Vc, Vd, Ve, Vf]



#Dissociation rates parameters (K)
Kda = {'name':'Kda','distribution':'loguniform', 'min':0.1, 'max':1000}
Keb = {'name':'Keb','distribution':'loguniform', 'min':0.1, 'max':1000}
Kfe = {'name':'Kfe','distribution':'loguniform', 'min':0.1, 'max':1000}
Kee = {'name':'Kee','distribution':'fixed','value':0.01}
Kce = {'name':'Kce','distribution':'loguniform', 'min':0.1, 'max':1000}
Kvd = {'name':'Kvd','distribution':'loguniform', 'min':1, 'max':1000}
Kub = {'name':'Kub','distribution':'loguniform', 'min':1, 'max':1000}
K_parameters = [Kda, Kub, Keb, Kvd, Kfe, Kee, Kce]


#Cooperativity parameters (n)
nvd = {'name':'nvd','distribution':'fixed', 'value':2}
nub = {'name':'nub','distribution':'fixed', 'value':1}
nda = {'name':'nda','distribution':'fixed', 'value':2}
nfe = {'name':'nfe','distribution':'fixed', 'value':5}
nee = {'name':'nee','distribution':'fixed', 'value':4}
neb = {'name':'neb','distribution':'fixed', 'value':4}
nce = {'name':'nce','distribution':'fixed', 'value':3}
n_parameters = [nvd,nub,nda,nfe,nee,neb,nce]


#protein degradation parameters (mu)
muASV = {'name':'muASV','distribution':'fixed', 'value':muASV_estimate/muASV_estimate}
muLVA = {'name':'muLVA','distribution': 'gaussian','mean':muLVA_estimate /muASV_estimate, 'noisetosignal':0.1}
mu_parameters = [muLVA,muASV]

#Diffusion parameters (D)
def Dr_(K1,K2,muU,muV,DU_,DV_):
    return (K2*DV_*muU)/(K1*DU_*muV)
Dr = {'name':'Dr','distribution':'loguniform', 'min':Dr_(K1,K2,muU,muV,DUmax,DVmin), 'max':Dr_(K1,K2,muU,muV,DUmin,DVmax)}
D_parameters = [Dr]


#Plot the distributions
plotDistributions=False
if plotDistributions == True:
    D_parameters_plotting = [Dr, Dr]
    nsamples=10000
    parameterTypeList = [ D_parameters_plotting  , V_parameters , K_parameters , mu_parameters , n_parameters]

    for parameterType in parameterTypeList:
        stackedDistributions = preLhs(parameterType)
        lhsDist = lhs(stackedDistributions,nsamples)
        lhsDist_df = pd.DataFrame(data = lhsDist, columns=[parameter['name'] for parameter in parameterType])
        plotDist(parameterType,lhsDist_df)
    print('Plotting distributions done!')

#Create parameter sets
createParams=True
if createParams == True:

    nsamples=100
    parameterDictList = D_parameters  + V_parameters + K_parameters + mu_parameters + n_parameters
    stackedDistributions = preLhs(parameterDictList)
    lhsDist = lhs(stackedDistributions,nsamples)
    lhsDist_df = pd.DataFrame(data = lhsDist, columns=[parameter['name'] for parameter in parameterDictList])
    pkl.dump(lhsDist_df, open('../../out/parameter_dataframes/literature_derived_parameters/df_circuit%r_variant%s_%rparametersets.pkl'%(circuit_n,variant,nsamples), 'wb'))
    print('Creating parameter sets done!')

#Create balanced parameter sets
createBalancedParams=True
if createBalancedParams == True:
    Km_list = ['Kda', 'Kub', 'Keb', 'Kvd', 'Kfe',  'Kce' ]
    KtoV = {'Kda': 'Vd', 'Kub': 'Va', 'Keb': 'Ve', 'Kvd': 'Vb', 'Kfe': 'Vf','Kce': 'Vc' }


    seed=0
    nsamples=100
    parameterDictList = D_parameters  + V_parameters + K_parameters + mu_parameters + n_parameters
    stackedDistributions = preLhs(parameterDictList)
    balancedDf = pd.DataFrame()
    semiBalancedDf = pd.DataFrame()
    notBalancedDf = pd.DataFrame()
    while len(balancedDf)<nsamples:
        lhsDist = lhs(stackedDistributions,nsamples, seed = seed, tqdm_disable = True)
        lhsDist_df = pd.DataFrame(data = lhsDist, columns=[parameter['name'] for parameter in parameterDictList])
       
       #check balance
        balanceList = []    
        for parID in lhsDist_df.index:
            par_dict = lhsDist_df.loc[parID].to_dict()
            balanceList.append(checkBalance(par_dict, Km_list, KtoV))
        lhsDist_df['balance'] = balanceList
        
        #separate 3df
        balancedDfPre = lhsDist_df[lhsDist_df['balance']=='Balanced']
        semiBalancedDfPre = lhsDist_df[lhsDist_df['balance']=='Semi balanced']
        notBalancedDfPre = lhsDist_df[lhsDist_df['balance']=='Not balanced']
        
        #concat to df
        if len(balancedDf)<nsamples:
            balancedDf = pd.concat([balancedDf, balancedDfPre], ignore_index=True)
        if len(semiBalancedDf)<nsamples:
            semiBalancedDf = pd.concat([semiBalancedDf, semiBalancedDfPre], ignore_index=True)
        if len(notBalancedDf)<nsamples:
            notBalancedDf = pd.concat([notBalancedDf, notBalancedDfPre], ignore_index=True)

        seed+=1
    
    pkl.dump(balancedDf[:nsamples], open('../../out/parameter_dataframes/literature_derived_parameters/df_circuit%r_variant%s_%rparametersets_balanced.pkl'%(circuit_n,variant,nsamples), 'wb'))
    pkl.dump(semiBalancedDf[:nsamples], open('../../out/parameter_dataframes/literature_derived_parameters/df_circuit%r_variant%s_%rparametersets_semibalanced.pkl'%(circuit_n,variant,nsamples), 'wb'))
    pkl.dump(notBalancedDf[:nsamples], open('../../out/parameter_dataframes/literature_derived_parameters/df_circuit%r_variant%s_%rparametersets_notbalanced.pkl'%(circuit_n,variant,nsamples), 'wb'))


    print('Creating balanced parameter sets done!')



