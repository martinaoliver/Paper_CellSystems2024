
#############

###paths#####
#############
import sys
import os

#############

from linear_stability_analysis_functions import big_turing_analysis_df, detailed_turing_analysis_dict, plot_all_dispersion, plot_highest_dispersion

# from randomfunctions import *

import pickle
import matplotlib.pyplot as plt
#######################
#########CODE##########
#######################


#%%
#Load parameter sets and dataframes
circuit_n='circuit1'
variant=1
n_samples=100
n_species=6 

df = pickle.load(open('../../out/parameter_dataframes/literature_derived_parameters/df_%s_variant%s_%rparametersets_balanced.pkl'%(circuit_n,variant,n_samples), 'rb'))

#%%


#Run analysis on 1M parameter sets
output_df = big_turing_analysis_df(df.iloc[:3],circuit_n, n_species, print_parID=False, tqdm_disable=False)

#%%
#Run analysis on a single parameter set
par_dict = df.iloc[0].to_dict()
out = detailed_turing_analysis_dict(par_dict, circuit_n,n_species,top_dispersion=1000,calculate_unstable=False,steadystate=False)


plot_all_dispersion(out[4][0],n_species, crop=100, top=300)
plt.show()
plt.close()

plot_highest_dispersion(out[4][0],crop = 70, top = 2000)
plt.show()

