#############
###paths#####
#############
import sys
import os

import sys
sys.path.append('../../input/equations/')
from class_circuit_eq import *
#############

import numpy as np
from scipy.sparse import spdiags, diags
from tqdm import tqdm
import copy
from scipy.linalg import solve_banded
import matplotlib.pyplot as plt
import numba
from numba import cuda, float32


def ADI_cellular_automata(par_dict,L,dx,J,T,dt,N, circuit_n, n_species,D,cell_matrix_record, daughterToMotherDictList,tqdm_disable=False,division_time_hours=1,stochasticity=0, seed=1, boundaryCoeff=1.5):
    
    parent_list = [circuit1]
    f = parent_list[circuit_n-1](par_dict, stochasticity=stochasticity)

    #spatial variables
    L_x=L;L_y=L;dy=dx;I=J

    #define diffussion species
    diffusing_species =np.nonzero(D)[0]
    nondiffusing_species = np.nonzero(D==0)[0]

    #alpha parameter required for ADI solver
    alpha = [D[n]*dt/(2.*dx*dy) for n in range(n_species)]


    #Define initial conditions and cell matrix
    U0 = []
    perturbation=0.001
    steadystates=[0.01]*n_species
    np.random.seed(seed)

    cell_matrix = cell_matrix_record[:,:,0]
    for index in range(n_species):
        U0.append(np.random.uniform(low=steadystates[index] - perturbation, high=steadystates[index] + perturbation, size=(I, J)))
    U0 = U0*cell_matrix

    #A matrix (right-hand side of Ax=b)
    def A(alphan):
        bottomdiag = [-alphan for j in range(J-1)]
        centraldiag = [1.+boundaryCoeff*alphan]+[1.+2.*alphan for j in range(J-2)]+[1.+boundaryCoeff*alphan]
        topdiag = [-alphan for j in range(J-1)]
        diagonals = [bottomdiag,centraldiag,topdiag]
        A = diags(diagonals, [ -1, 0,1]).toarray()
        return A
    def diagonal_form(a, upper = 1, lower= 1):
        """
        a is a np square matrix
        this function converts a square matrix to diagonal ordered form
        returned matrix in ab shape which can be used directly for scipy.linalg.solve_banded
        """
        n = a.shape[1]
        assert(np.all(a.shape ==(n,n)))

        ab = np.zeros((2*n-1, n))

        for i in range(n):
            ab[i,(n-1)-i:] = np.diagonal(a,(n-1)-i)

        for i in range(n-1):
            ab[(2*n-2)-i,:i+1] = np.diagonal(a,i-(n-1))

        mid_row_inx = int(ab.shape[0]/2)
        upper_rows = [mid_row_inx - i for i in range(1, upper+1)]
        upper_rows.reverse()
        upper_rows.append(mid_row_inx)
        lower_rows = [mid_row_inx + i for i in range(1, lower+1)]
        keep_rows = upper_rows+lower_rows
        ab = ab[keep_rows,:]


        return ab
    ab_list = [diagonal_form(A(alphan)) for alphan in alpha]
    A_list = [A(alphan) for alphan in alpha]


    #b vector (left-hand side of Ax=b)
    @numba.jit(nopython=True)
    def b(axis,ij,alphan,Un):
        b_t_stencil = np.array( [0] + [(1-boundaryCoeff*alphan)] + [alphan])
        b_c_stencil = np.array( [alphan] + [(1-2*alphan)] + [alphan])
        b_b_stencil = np.array( [alphan] + [(1-boundaryCoeff*alphan)] + [0])
        
        b = np.zeros(J)
        if axis == 'y':
            i = ij
            if i > 0 and i < I-1:
                for j in range(0,J):
                    # ux_three = [Un[j,i-1], Un[j,i], Un[j,i+1]]
                    ux_three = np.array( [Un[j,i-1]] +  [Un[j,i]]+  [Un[j,i+1]])
                    sub_b = sum(ux_three*b_c_stencil)
                    b[j] = sub_b

            if i == 0:
                for j in range(0,J):
                    ux_three = np.array([0] +[Un[j,i]]+[Un[j,i+1]])
                    sub_b = sum(ux_three*b_t_stencil)
                    b[j] = sub_b

            if i == I-1:
                for j in range(0,J):
                    ux_three = np.array([Un[j,i-1]] + [Un[j,i]] + [0])
                    sub_b = sum(ux_three*b_b_stencil)
                    b[j] = sub_b


        if axis == 'x':
            j = ij
            if j > 0 and  j < J-1:
                for i in range(0,I):
                    uy_three = np.array([Un[j-1,i]]+ [Un[j,i]] + [Un[j+1,i]])
                    sub_b = sum(uy_three*b_c_stencil)
                    b[i] = sub_b

            if j == 0:
                for i in range(0,I):
                    uy_three = np.array([0] +[Un[j,i]] +  [Un[j+1,i]])
                    sub_b = sum(uy_three*b_t_stencil)
                    b[i] = sub_b

            if j == J-1:
                for i in range(0,I):
                    uy_three = np.array([Un[j-1,i]] + [Un[j,i]] + [0])
                    sub_b = sum(uy_three*b_b_stencil)
                    b[i] = sub_b

        return b




    #create initial [] matrices and record arrays
    U = U0.copy()
    U_record = []
    for species_index in range(n_species):
        U_record.append(np.zeros([J, I, T])) #DO NOT SIMPLIFY TO U_record = [np.zeros([J, I, T])]*n_species


    #define parameters for divisio frequency
    division_time_hours=1 #cells consider division every x hours
    divisionTimeUnits=int(division_time_hours/dt) #cells consider division every x timeunits. If dt=0.1, x=10
    A_inv = [np.linalg.inv(a) for a in A_list]

    for ti in tqdm(range(0,N), disable = tqdm_disable):

        #define current cell matrix for time ti
        cell_matrix = cell_matrix_record[:,:,ti]

        #First step: solve in y direction from n -> n+1/2        
        U_half = U.copy()
        f0 = f.dudt_growth(U,cell_matrix)
        for i in range(I):
            for n in diffusing_species:
                U_half[n][:,i] = A_inv[n].dot(b('y',i,alpha[n],U[n])+  f0[n][:,i]*(dt/2)) # Dot product with inverse rather than solve system of equations
                # U_half[n][:,i] = solve_banded((1, 1), ab_list[n], b('y',i,alpha[n],U[n]) +  f0[n][:,i]*(dt/2)) #CN step in one dimension to get banded(tridiagonal) A matrix
            for n in nondiffusing_species: #Species with D=0, no CN included, basic euler
                U_half[n][:,i] =  U[n][:,i] + f0[n][:,i]*(dt/2)

        #Second step: solve in x direction from n+1/2 -> n+1
        U_new = U_half.copy()
        f1 = f.dudt_growth(U_half,cell_matrix)
        for j in range(J):
            for n in diffusing_species:
                U_new[n][j,:] = A_inv[n].dot(b('x',j,alpha[n],U_half[n])+  f1[n][j,:]*(dt/2)) # Dot product with inverse rather than solve system of equations

                # U_new[n][j,:] = solve_banded((1, 1), ab_list[n], b('x',j,alpha[n],U_half[n]) + f1[n][j,:]*(dt/2))
            for n in nondiffusing_species:
                U_new[n][j,:] =  U_half[n][j,:] + f1[n][j,:]*(dt/2)
       
        #if at this time, division occurs: add mother [] to daughter cells
        if (ti%divisionTimeUnits==0):
            daughterToMotherDict = daughterToMotherDictList[ti-1]
            for newCell,oldCell in daughterToMotherDict.items():
                U_new[:,newCell[0],newCell[1]] = U_new[:,oldCell[0],oldCell[1]]

        hour = ti / (N / T)

        if hour % 1 == 0:  #only consider recording at unit time (hour)
            #append results into top_array for records
            for species_index in range(n_species):
                U_record[species_index][:, :, int(hour)] = U_new[species_index] #issue in this line
        
        for n in range (n_species):
            if any(x<0 for x in U_new[n].flatten()):
                print('negative')

        U = U_new.copy()

    return U_record, U
