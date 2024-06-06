#%%
#############
###paths#####
#############

import pickle
import time
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from colony_mask_creation import run_cellular_automata_colony
from ADI_cellular_automata_functions import ADI_cellular_automata
from numerical_plotting_functions import plot_redgreen_contrast



#%%
#############
###execution parameters#####
#############
shape = 'ca'
circuit_n=1;variant=1;n_species=6
n_samples = 100
balance = 'balanced'
output_directory = f"../../out/numerical_results/circuit{circuit_n}_variant{variant}_{balance}" #folder to save numerical results
Path(output_directory).mkdir(parents=True, exist_ok=True) #create folder if it does not exist


save_figure = True
tqdm_disable = False #disable tqdm


# open parameter dictionaries
df= pickle.load( open( "../../out/parameter_dataframes/literature_derived_parameters/df_circuit1_variant1_100parametersets_balanced.pkl", "rb"))


#slowgrowth
L=20; dx =0.1; J = int(L/dx)
T =100; dt = 0.02; N = int(T/dt)
boundaryCoeff = 2
division_time_hours=0.5
p_division=0.38;seed=1
# 
#mediumgrowth
L=20; dx =0.1; J = int(L/dx)
T =50; dt = 0.02; N = int(T/dt)
boundaryCoeff = 1
division_time_hours=0.5
p_division=1;seed=1

# # fastgrowth
L=20; dx =0.1; J = int(L/dx)
T =25; dt = 0.02; N = int(T/dt)
boundaryCoeff = 1
division_time_hours=0.2
p_division=0.7;seed=1


shape = 'ca'
x_gridpoints=int(1/dx)

try:
    cell_matrix_record = pickle.load( open("../../out/cellular_automata_masks/caMask_seed%s_pdivision%s_L%s_J%s_T%s_N%s.pkl"%(seed,p_division,L,J,T,N), "rb" ) )
    daughterToMotherDictList = pickle.load( open("../../out/cellular_automata_masks/caMemory_seed%s_pdivision%s_L%s_J%s_T%s_N%s.pkl"%(seed,p_division,L,J,T,N), "rb" ) )
    print('Cellular automata mask already exists. Loaded.')

except:
    #file does not exist
    FileNotFoundError
    print('Cellular automata mask does not exist. Running cellular automata.')
    run_cellular_automata_colony(L=L,dx=dx, T=T ,dt=dt, divisionTimeHours=division_time_hours, p_division=p_division, plot1D=True, plotScatter=True, plotVolume=True)
    # run_cellular_automata_colony(L=L,dx=dx, T=T, dt=dt, division_time_hours=division_time_hours, p_division=p_division, plot1D=True, plotScatter=True)
    cell_matrix_record = pickle.load( open("../../out/cellular_automata_masks/caMask_seed%s_pdivision%s_L%s_J%s_T%s_N%s.pkl"%(seed,p_division,L,J,T,N), "rb" ) )
    daughterToMotherDictList = pickle.load( open("../../out/cellular_automata_masks/caMemory_seed%s_pdivision%s_L%s_J%s_T%s_N%s.pkl"%(seed,p_division,L,J,T,N), "rb" ) )
    


filename= lambda parID: 'circuit%r_variant%s_bc%s_%s_ID%r_L%r_J%r_T%r_N%r'%(circuit_n,variant,boundaryCoeff, shape,parID,L,J,T,N)
parID=1


#%%
test = True
tqdm_disable=False
if test==True:
    print('test')
    T=1;N =1
    tqdm_disable=False


print('parID = ' + str(parID))
par_dict = df.loc[parID].to_dict()

D = np.zeros(n_species)
Dr = float(par_dict['Dr'])
D[:2] = [1,Dr ]


st = time.time()
U_record,U_final =  ADI_cellular_automata(par_dict,L,dx,J,T,dt,N, circuit_n, n_species,D,cell_matrix_record, daughterToMotherDictList,tqdm_disable=tqdm_disable,division_time_hours=division_time_hours, stochasticity=0, seed=1, boundaryCoeff=boundaryCoeff)
elapsed_time = time.time() - st
print('Execution time:', time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))

pickle.dump(U_final, open("%s/2Dfinal_%s.pkl"%(output_directory,filename(parID)), "wb" ) )
pickle.dump(U_record, open("%s/2Drecord_%s.pkl"%(output_directory,filename(parID)), 'wb'))


rgb = plot_redgreen_contrast(U_final,L,parID=filename(parID),filename = filename(parID), path = output_directory,scale_factor=x_gridpoints,save_figure=True)
