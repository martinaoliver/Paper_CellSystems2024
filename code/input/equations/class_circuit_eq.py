import numpy as np
from numpy.random import normal

# - This file contains the definitions for the genetic circuit used. The circuit is composed of different
# molecular interactions.


# - The system is defined with differential equations for every specie where hill functions are used.
# - Concentrations of the morphogens and model parameters are inputs to these classes.

# - Every class contains (1) Differential equations and (2) jacobian matrix for the system.
# - The jacobian matrix is calculated in advance and defined here so it doesn't have to be
#calculated every time a parameter set is analysed (saves computational power).

class hill_functions():

    def __init__(self, par_dict):
        for key, value in par_dict.items():
            setattr(self, key, value)

    def noncompetitiveact(self, X, km,n):
        # act = ((X / km) ** (n)) / (1 + (X / km) ** (n))
        act = (1 / (1 + (km / (X + 1e-8)) ** (n)))
        return act

    def noncompetitiveinh(self, X, km,n):
        inh = 1 / (1 + (X / (km + 1e-8) ) ** (n))
        return inh


    def noncompetitivediffact(self, X, km,n, kdiff,mudiff):
        act = (1 / (1 + ((mudiff*km) / (kdiff*X)) ** (n)))
        return act
 




#Dimensionless equations with 8to6 transition
class circuit1(hill_functions):

    def __init__(self,par_dict,stochasticity=0):
        for key,value in par_dict.items():
            setattr(self,key,value)
        setattr(self, 'stochasticity', stochasticity)


    def dAdt_f(self,species_list, wvn=0):
        A,B,C,D,E,F = species_list
        dadt= 1 + self.Va*self.noncompetitiveinh(D,self.Kda,self.nda) -A - A*wvn**2
        if self.stochasticity ==1:
            dadt+=dadt*normal(0,0.05,1)
        return dadt


    def dBdt_f(self,species_list, wvn=0):
        A,B,C,D,E,F = species_list
        dbdt= self.muLVA*(1 + self.Vb*self.noncompetitiveact(A,self.Kub, self.nub)*self.noncompetitiveinh(E,self.Keb, self.neb) - B ) -  B*self.Dr*wvn**2
        if self.stochasticity ==1:
            dbdt+=dbdt*normal(0,0.05,1)
        return dbdt


    def dCdt_f(self,species_list):
        A,B,C,D,E,F = species_list
        dcdt= self.muLVA*(1 + self.Vc*self.noncompetitiveinh(D,self.Kda,self.nda) - C ) 
        if self.stochasticity ==1:
            dcdt+=dcdt*normal(0,0.05,1)
        return dcdt

    def dDdt_f(self,species_list):
        A,B,C,D,E,F = species_list
        dddt= self.muLVA*(1 + self.Vd*self.noncompetitiveact(B,self.Kvd,self.nvd) - D ) 
        if self.stochasticity ==1:
            dddt+=dddt*normal(0,0.05,1)
        return dddt

    def dEdt_f(self,species_list):
        A,B,C,D,E,F = species_list
        dedt= self.muLVA*(1 + self.Ve*self.noncompetitiveinh(C,self.Kce,self.nce)*self.noncompetitiveinh(F,self.Kfe,self.nfe)*self.noncompetitiveact(E,self.Kee,self.nee) - E ) 
        if self.stochasticity ==1:
            dedt+=dedt*normal(0,0.05,1)
        return dedt
        
    def dFdt_f(self,species_list):
        A,B,C,D,E,F = species_list
        dfdt= self.muLVA*(1 + self.Vf*self.noncompetitiveact(B,self.Kvd,self.nvd) - F ) 
        if self.stochasticity ==1:
            dfdt+=dfdt*normal(0,0.05,1)
        return dfdt

    function_list = [dAdt_f,dBdt_f,dCdt_f,dDdt_f,dEdt_f,dFdt_f]
    
    def dudt_growth(self,U, cell_matrix):
        function_list = [self.dAdt_f(U),self.dBdt_f(U),self.dCdt_f(U), self.dDdt_f(U),self.dEdt_f(U),self.dFdt_f(U)]
        dudt = [eq*cell_matrix for eq in function_list]
        return dudt
    def dudt(self,U):
        dudt = [self.dAdt_f(U),self.dBdt_f(U),self.dCdt_f(U), self.dDdt_f(U),self.dEdt_f(U),self.dFdt_f(U)]
        return dudt

    def getJacobian(self,x,wvn):

        A,B,C,D,E,F = x

        JA = [-wvn**2 - 1, 0, 0, -self.Va*self.nda*(D/self.Kda)**self.nda/(D*((D/self.Kda)**self.nda + 1)**2), 0, 0]
        JB = [self.Vb*self.muLVA*self.nub*(self.Kub/A)**self.nub/(A*((self.Kub/A)**self.nub + 1)**2*((E/self.Keb)**self.neb + 1)), -self.Dr*wvn**2 - self.muLVA, 0, 0, -self.Vb*self.muLVA*self.neb*(E/self.Keb)**self.neb/(E*((self.Kub/A)**self.nub + 1)*((E/self.Keb)**self.neb + 1)**2), 0]
        JC = [0, 0, -self.muLVA, -self.Vc*self.muLVA*self.nda*(D/self.Kda)**self.nda/(D*((D/self.Kda)**self.nda + 1)**2), 0, 0]
        JD = [0, self.Vd*self.muLVA*self.nvd*(self.Kvd/B)**self.nvd/(B*((self.Kvd/B)**self.nvd + 1)**2), 0, -self.muLVA, 0, 0]
        JE = [0, 0, -self.Ve*self.muLVA*self.nce*(C/self.Kce)**self.nce/(C*((C/self.Kce)**self.nce + 1)**2*((self.Kee/E)**self.nee + 1)*((F/self.Kfe)**self.nfe + 1)), 0, self.muLVA*(-1 + self.Ve*self.nee*(self.Kee/E)**self.nee/(E*((C/self.Kce)**self.nce + 1)*((self.Kee/E)**self.nee + 1)**2*((F/self.Kfe)**self.nfe + 1))), -self.Ve*self.muLVA*self.nfe*(F/self.Kfe)**self.nfe/(F*((C/self.Kce)**self.nce + 1)*((self.Kee/E)**self.nee + 1)*((F/self.Kfe)**self.nfe + 1)**2)]
        JF = [0, self.Vf*self.muLVA*self.nvd*(self.Kvd/B)**self.nvd/(B*((self.Kvd/B)**self.nvd + 1)**2), 0, 0, 0, -self.muLVA]
        return np.array([JA, JB, JC, JD, JE, JF])
