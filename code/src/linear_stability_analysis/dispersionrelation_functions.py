import sys
sys.path.append('../../input/equations/')
from class_circuit_eq import *

import numpy as np
from numpy import linalg as LA
from numpy import unravel_index
#This class can return the jacobian which is stored in circuit*_eq class.
class jacobian():
    def __init__(self,par_dict, circuit_n):
        for key, value in par_dict.items():
            setattr(self, key, value)
        setattr(self, 'circuit_n', circuit_n)

        # self.parent_list = [circuit1_eq, circuit2_eq, circuit3_eq, circuit4_eq, circuit5_eq, circuit6_eq, circuit7_eq]
        self.parent_list = {'circuit1':circuit1}

    def getJacobian(self,x,wvn):  # circuit1_eq
        # return self.parent_list[self.circuit_n-1].getJacobian(self,x,wvn)
        return self.parent_list[self.circuit_n].getJacobian(self,x,wvn)


def calculate_dispersion(par_dict,circuit_n, x,top_dispersion):
    n_species = len(x)

    # - Define which wavenumbers will be analysed. L = 100 in this case (100mm) (The experimental system is a 10cm plate)
    # so wavelengths bigger than that are of no interest.
    # - In this case we will sample 5000+1 different wavenumbers. If you sample less you might not find your turing instability.
    # If you sample more, is more computationally expensive.

    # wvn_list = np.array(list(range(0,5000+1)))*np.pi/100
    wvn_list = np.array(list(range(0,top_dispersion+1)))*np.pi/100
    count = 0
    eigenvalues = np.zeros((len(wvn_list),n_species) ,dtype=np.complex_)

    for wvn in wvn_list:
        jac = jacobian(par_dict, circuit_n).getJacobian(x, wvn) #obtain jacobian for corresponding system. This time with a determined wvn.
        eigenval, eigenvec = LA.eig(jac) #calculate the eigenvalues of the jacobian with diffusion
        # sort eigenvalues so the one with the instability is at position -1.
        idx = np.argsort(eigenval) #np.argsort
        eigenval= eigenval[idx] #orders eigenvalues for each k.
        eigenvalues[count]  = eigenval
        count +=1

    #find the wavenumber with the highest instability
    max_index = unravel_index(np.real(eigenvalues).argmax(), np.real(eigenvalues).shape)[0]
    max_wvn = wvn_list[max_index]
    if max_wvn == 0:
        estimated_wvl = np.nan
    else:
        estimated_wvl = 2*np.pi/(max_wvn)


    return eigenvalues, estimated_wvl

def stability_no_diffusion(eigenvalues):

    eigenvalues_ss = eigenvalues[0,-1]

    #oscillations in ss
    if np.iscomplex(eigenvalues_ss):
        complex_ss=True
    else:
        complex_ss=False

    #oscillations overall
    if np.iscomplex(eigenvalues).any():
        complex_dispersion=True
    else:
        complex_dispersion=False

    #stability in  ss
    if eigenvalues_ss > 0:
        stability_ss = 'unstable'
    elif eigenvalues_ss == 0:
        stability_ss = 'lyapunov stable'
    elif eigenvalues_ss < 0:
        stability_ss = 'stable'

    #classification of ss
    if complex_ss==False:
        if stability_ss == 'stable':
            ss_class =  'stable point'
        elif stability_ss == 'lyapunov stable':
            ss_class =  'neutral point'
        elif stability_ss == 'unstable':
            ss_class =  'unstable point'
    if complex_ss==True:
        if stability_ss == 'stable':
            ss_class =  'stable spiral'
        elif stability_ss == 'lyapunov stable':
            ss_class =  'neutral center'
        elif stability_ss == 'unstable':
            ss_class =  'unstable spiral'

    return ss_class, complex_ss, stability_ss, complex_dispersion

def stability_diffusion(eigenvalues, ss_class, complex_ss, stability_ss, complex_dispersion):
    maxeig = np.amax(eigenvalues) #highest eigenvalue of all wvn's and all the 6 eigenvalues.
    maxeig_real = maxeig.real #real part of maxeig
    if stability_ss == 'stable' or stability_ss == 'neutral':#turing I, turing I oscillatory, turing II, Unstable
        if maxeig_real <= 0:
            system_class = 'simple stable'
        elif maxeig_real > 0: #turing I, turing I oscillatory, turing II
            if np.any(eigenvalues[-1,:] == maxeig):
                system_class = 'turing II'
            elif np.all(eigenvalues[-1,:] != maxeig): #highest instability does not appear with highest wavenumber)
                if np.any(eigenvalues.imag[:,-1]>0): #if the highest eigenvalue contains imaginary numbers oscillations might be found
                    system_class = 'turing I oscillatory'
                elif np.all(eigenvalues.imag[:,-1]<=0): #if the highest eigenvalue does not contain imaginary numbers
                    system_class = 'turing I'
                else:
                    system_class = 'unclassified1'
            else:
                system_class = 'unclassified2'
        else:
            system_class = 'max_eig_real other' 



    elif stability_ss == 'unstable': #fully unstable or hopf
        if complex_dispersion==False:
                system_class = 'simple unstable'

        # elif complex_dispersion==True and eigenvalues.imag[-1,-1]==0: #hopf
        elif complex_dispersion==True: 
            
            # and eigenvalues.imag[-1,-1]==0: #hopf
            #sign changes
            real_dominant_eig = eigenvalues.real[:,-1]
            indexZeros = np.where(np.diff(np.sign(real_dominant_eig)))[0]
            sign_changes = len(indexZeros)
            if sign_changes>0:
                if np.iscomplex([eigenvalues[:,-1][index] for index in indexZeros]).any():
                    
                    hopf=True
                else:
                    hopf=False

            if sign_changes == 1:
                if hopf==True:
                    system_class = 'hopf'
                if hopf==False:
                    system_class = 'complex unstable'
            elif sign_changes > 1:
                if hopf==True:
                    if np.any(eigenvalues[-1,-1] == maxeig_real):
                        system_class = 'turing II hopf'
                    elif np.all(eigenvalues[-1,-1] != maxeig_real): #highest instability does not appear with highest wavenumber)
                        system_class = 'turing I hopf'
                    else:
                        system_class='Hopf other'
                else:
                    system_class = 'turing semi-hopf'

            else:
                system_class = 'Zero sign changes'
        else:
            system_class = 'Complexity other'
    else:
        system_class = 'Stability other'




    return system_class, maxeig


def dispersionrelation(par_dict,steadystate_values_ss_n,circuit_n,top_dispersion):

    x = steadystate_values_ss_n

    #calculate dispersion and obtain eigenvalues
    eigenvalues, estimated_wvl = calculate_dispersion(par_dict,circuit_n, x,top_dispersion)

    #Classify equilibrium point (no diffusion)
    ss_class, complex_ss, stability_ss, complex_dispersion = stability_no_diffusion(eigenvalues)

    system_class, maxeig= stability_diffusion(eigenvalues, ss_class, complex_ss, stability_ss, complex_dispersion)


    #find the wavenumber with highest eigenvalue
    max_index = unravel_index(eigenvalues.argmax(), eigenvalues.shape)[0]


    return ss_class, system_class, eigenvalues, maxeig, estimated_wvl,complex_dispersion
