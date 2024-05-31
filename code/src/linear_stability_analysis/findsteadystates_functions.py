import sys
sys.path.append('../../input/equations/')
from class_circuit_eq import *

from lhs_functions import *

#Function performing newton_raphson.
def newton_raphson(f, x_guess,equations_par_dict, max_num_iter=15, tolerance=0.000001, alpha=1):
    '''
    Function for representing a Newton-Raphson iteration for multidimensional systems of equations
    :param f: function class that must define the following methods:
        - numDims(): Method that returns an integer number of variables in the system of equations
        - getJacobian(np.ndarray): Method to compute the Jacobian of the system of equations at the current root estimate.The output is an n by n matrix where n is the number of variables in the system of equations
		- __call__(np.ndarray): Method to make this class act like a function operating on some input x
    :param x_guess: an initial guess for the Newton-Raphson iteration
    :param max_num_iter: a maximum number of iterations that will be taken
    :param tolerance: a tolerance that will stop the sequence once the error drops below it
    :param alpha: A coefficient that can tune the Newton-Raphson stepsize. Recommend setting alpha <= 1.
    :return: A tuple with the root estimate, final error for the root, and the number of iterations it took
    '''
    # set the initial guess
    x = x_guess

    # compute function value at initial guess
    fx = f(x)

    # define the initial value for the error and the starting iteration count
    err = np.linalg.norm(fx)
    iter = 0
    # perform the Newton-Raphson iteration algo
    while err > tolerance and iter < max_num_iter and np.all(x!=0):
        # perform newton step

        x = x - alpha*np.linalg.solve(equations_par_dict.getJacobian(x),fx)

        # update the function value at the new root estimate
        fx = f(x)

        # compute the current root error
        err = np.linalg.norm(fx)

        # update the iteration counter
 
        iter = iter + 1

    if err < tolerance:
        if sum(item < 0 for item in x) == 0 :
            return (x, err, 0)
    else:
        return (x,err,1)
class newtonraphson_equations(hill_functions):

    def __init__(self,par_dict,circuit_n, stochasticity=1):
        for key,value in par_dict.items():
            setattr(self,key,value)
        setattr(self, 'circuit_n', circuit_n)
        setattr(self, 'stochasticity', 1)

        self.parent_list = {'circuit1':circuit1}
    
    def diff_equations(self, x):
        n=0
        # circuit = self.parent_list[self.circuit_n-1]
        circuit = self.parent_list[self.circuit_n]
        function_list = circuit.function_list
        f = np.zeros((len(function_list),))

        for function in function_list:
            
            f[n]=circuit.function_list[n](self,x)
            n +=1

        return f

    def getJacobian(self,x):
        # return self.parent_list[self.circuit_n-1].getJacobian(self,x,0)
        
        return self.parent_list[self.circuit_n].getJacobian(self,x,0)

def newtonraphson_run(par_dict,initial_conditions, circuit_n):
    # - The equations of the system are defined through this command which calls a class containing system equations.
    # The circuit_n indicates which system of equations you want to obtain. There are different circuits defined.
    # The par_dict will be taken by this class to insert parameter values into the equations.
    equations_par_dict = newtonraphson_equations(par_dict,circuit_n) #system definition: equations with parameters defined using par_dict
    f = equations_par_dict.diff_equations

    # - This loop performs a newton raphson analysis on the defined system above over all initial conditions selected.
    clusteredsteadystates = np.zeros(np.shape(initial_conditions))
    countlist = 0  # count elements added to list
    clusteredcountlist = 0

    for n in range(len(initial_conditions)):
        #this command performs newton raphson on a specific initial condition with the defined system (equations, parameters).
        xn = newton_raphson(f,initial_conditions[n],equations_par_dict)

        if xn == None: #no steady state found at that initial condition
            pass

        elif xn[2]==0: #steady state found at that initial condition

            steadystate = np.round(xn[0],4)
            if countlist == 0: #Always add to add clusteredsteadystates in the first iteration of this for loop.
                clusteredsteadystates[0,:] = steadystate
                clusteredcountlist +=1 #keeps count of number of steady states added to clusteredsteadystates

            if countlist > 0: #if not the first iteration, the steady state it must be compared against previous steady states stored to see if its the same
                logiclist = []
                for i in range(clusteredcountlist):
                    logiclist.append(np.allclose(clusteredsteadystates[i], steadystate, rtol=10**-2, atol=0)) #PROCEED IF NO TRUES FOUND
                if not True in logiclist: #no similar steady states previously found
                    clusteredsteadystates[clusteredcountlist] = steadystate
                    clusteredcountlist +=1
            countlist +=1

    # - Removes any zeros left in the bottom rows of the clusteredsteadystates list.
    # Some zeros appear if same steady state found multiple times
    if clusteredsteadystates[-1,0] == 0:
        clusteredsteadystates = clusteredsteadystates[~np.all(clusteredsteadystates == 0, axis=1)]

    # - Remove any steady states with zero to avoid errors down the line
    # clusteredsteadystates = clusteredsteadystates[np.all(clusteredsteadystates != 0, axis=1)]
    clusteredsteadystates[clusteredsteadystates==0]=1e-300 #set zeros to practically zero to avoid errors
    return clusteredsteadystates


#input a dictionary with the parameters and returns
# (1) a list with where each element is a steady state (within every steady state then has a value for each molecular specie)
# (2) the number of steady states.
def findsteadystates(par_dict, circuit_n, n_species, n_initial_conditions = 500):
    n_species = 6
    # - Define initial conditions that will be tested to find all possible steady states (higher n_initial_conditions, higher chance of finding all)
    # - The initial conditions sampled are found using latin hypercube sampling (an efficient method  to cover a wide area of the
    #initial condition parameter space with less samples).
    initial_conditions = lhs_initial_conditions(n_initial_conditions,n_species)
    
    # - Calls the function that evaluates the steady state of our system (with the parameters indicated in par_dict) for
    #every initial condition.
    # - It returns an array with all the 'unique' steady states obtained. From two different initial conditions you can reach
    # the same steady state. Therefore 'equal' steady states are clustered and only
    # steady states that are different are present in this array.
    clusteredsteadystates = newtonraphson_run(par_dict,initial_conditions, circuit_n)


    #transform array into aa list with the different steady states
    steadystatelist = []
    number_steadystates = len(clusteredsteadystates)
    if number_steadystates==0:
        print('no steady states')
    for n in range(number_steadystates):
        ss = clusteredsteadystates[n]
        steadystatelist.append(ss)


    return steadystatelist, number_steadystates
