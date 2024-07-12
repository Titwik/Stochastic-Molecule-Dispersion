# MT5855 Assignment

# import relevant modules
import math
import numpy as np
import matplotlib.pyplot as plt
import time
######################################################################################
# Q3

# define the function
def gillespie():
    
    # set times
    T = 300        
    t = [0]
    t_current = t[0]
    
    # set constants
    L = 1                   # length of interval
    K = 40                  # number of compartments
    D = 10 ** (-4)          # diffusion constant
    N = 10000               # number of molecules in total 
    h = L/K                 # length of each sub-interval
    d = D/(h**2)            # 'chemical' rate constant
    
    # create array to store the number of molecules in each compartment
    molecules = np.zeros(K, dtype = int)
    molecules[15], molecules[16] = 5000, 5000      # 5000 molecules in 16th and 17th compartments
    
    # create array for propensities
    prop = np.zeros(K)
    
    # initialise propensities for 16th and 17th compartments
    prop[15] = molecules[15] * d
    prop[16] = molecules[16] * d
    
    # total propensity
    prop_total = 2*N*d - prop[0] - prop[K-1]
    
    # start the algorithm
    while t_current < T:
        
        # generate tau, and compute the current time
        tau = np.random.exponential(1/prop_total)
        t_current += tau
        t.append(t_current)
        
        # store the values of the first and last compartment's molecule numbers
        first = molecules[0]
        last = molecules[K-1]
        
        # generate a random uniformly distrubuted number
        u = np.random.uniform()
        
        # compute the compartment where molecule jumps to the right
        if u < np.sum(prop[0:(K-1)])/prop_total:
            
            # find which compartment the jump happened in
            for j in range(1, K):
                previous_prop = np.sum(prop[:j-1])
                current_prop = np.sum(prop[:j])
                
                # conditions for jumping to the right
                if u >= previous_prop/prop_total and u < current_prop/prop_total:
                    
                    # jth compartment loses a molecule 
                    # (j+1) compartment gains a molecule
                    molecules[j-1] -= 1
                    molecules[j] += 1
                    
                    # recompute propensities for compartment j and j+1
                    prop[j-1] = molecules[j-1] * d
                    prop[j] = molecules[j] * d
                    
                    # break out of the loop once molecule numbers have changed
                    break
        
        # compute the compartment where molecule jumps to the left
        elif u < (np.sum(prop[0:(K-1)]) + np.sum(prop[1:K]))/prop_total:
            
            # find which compartment the jump happened in
            for j in range(2,K+1):
                previous_prop = np.sum(prop[0:K-1]) + np.sum(prop[1:j-1])
                current_prop = np.sum(prop[0:K-1]) + np.sum(prop[1:j])
                
                # conditions for jumping to the left
                if u >= previous_prop/prop_total and u < current_prop/prop_total:
                    molecules[j-1] -= 1
                    molecules[j-2] += 1
                    
                    # recompute propensities for compartment j and j-1
                    # jth compartment loses a molecule
                    # (j-1) compartment gains a molecule
                    prop[j-1] = molecules[j-1] * d
                    prop[j-2] = molecules[j-2] * d
                    
                    # break out of the loop once molecule numbers have changed
                    break
        
        # compute the new total propensity if a reaction occured
        # in the first or last compartment
        if molecules[0] != first or molecules[K-1] != last:
            prop_total = 2*N*d - prop[0] - prop[K-1]            
    
    # create a histogram visualizing the number of molecules
    # in each compartment
    x_values = np.arange(1, K + 1)
    plt.bar(x_values, molecules)
    plt.xlabel('Compartments')
    plt.ylabel('Number of Molecules')
    plt.title('Q3) Compartment based Diffusion')
    plt.show()
    
######################################################################################
# Q5

# define the function 
def schnakenberg():
    
    # set times
    T = 1800              # 30 mins = 1800 seconds
    t_current = 0
    times = [t_current]
    
    # set constants
    L = 1                 # length of interval
    K = 40                # number of compartments
    h = L/K               # length of each compartment
    D_A = 10 ** (-5)      # species A diffusion constant
    D_B = 10 ** (-3)      # species B diffusion constant
    d_A = D_A/(h**2)      # A's rate of diffusion
    d_B = D_B/(h**2)      # B's rate of diffusion
    
    # reaction constants
    k1 = 10**(-4)    # 2A + B -> 3A
    k2 = 0.1         # 0 -> A
    k3 = 0.02        # A -> 0
    k4 = 0.3         # 0 -> B 
    
    # initialise number of molecules
    molecules_A = np.full(K,200,dtype = int)
    molecules_B = np.full(K,75,dtype = int)
        
    # create an array to store the diffusion propensities
    prop_diff_A = molecules_A * d_A
    prop_diff_B = molecules_B * d_B
    
    # reaction propensities
    k_1 = molecules_A * (molecules_A - 1) * molecules_B * k1
    k_2 = np.full(K, k2)
    k_3 = molecules_A * k3
    k_4 = np.full(K, k4)
    
    # compute the total propensity of the system
    # Not using 2*N*d - a_1 - a_k as total molecule numbers are not conserved
    # due to production and degradation reactions
    prop_total_system = (np.sum(prop_diff_A[0:K-1]) + np.sum(prop_diff_A[1:K]) +
                         np.sum(prop_diff_B[0:K-1]) + np.sum(prop_diff_B[1:K]) +
                         np.sum(k_1) + np.sum(k_2) + np.sum(k_3) + np.sum(k_4))
    
    # start the algorithm
    while t_current < T:
        
        # generate tau (time until next reaction) and compute the current time
        tau = np.random.exponential(1/prop_total_system)
        t_current += tau
        times.append(t_current)
        
        # generate a random uniformly distributed number
        u = np.random.uniform()
        
        # compute if A_i jumps to the right
        if u < np.sum(prop_diff_A[0:(K-1)])/prop_total_system:
            
            # find the compartment the jump occurs in 
            for j in range(1,K):
                previous_prop_A = np.sum(prop_diff_A[0:j-1])
                current_prop_A = np.sum(prop_diff_A[0:j])
                
                # conditions to jump to the right
                if u >= previous_prop_A/prop_total_system and u < current_prop_A/prop_total_system:
                    
                    # jth compartment loses a molecule 
                    # (j+1) compartment gains a molecule
                    molecules_A[j-1] -= 1
                    molecules_A[j] += 1
                    
                    # recompute propensities 
                    prop_diff_A[j-1] = molecules_A[j-1] * d_A                    
                    k_1[j-1] = molecules_A[j-1] * (molecules_A[j-1] - 1) * molecules_B[j-1] * k1
                    k_3[j-1] = molecules_A[j-1] * k3
                    
                    prop_diff_A[j] = molecules_A[j] * d_A
                    k_1[j] = molecules_A[j] * (molecules_A[j] - 1) * molecules_B[j] * k1
                    k_3[j] = molecules_A[j] * k3
                    
                    # break out of the loop 
                    break
                
        # compute if A_i jumps to the left
        elif u < (np.sum(prop_diff_A[0:(K-1)]) + np.sum(prop_diff_A[1:K]))/prop_total_system:
            
            # find the compartment the jump occured in
            for j in range(2, K+1):
                previous_prop_A = np.sum(prop_diff_A[0:K-1]) + np.sum(prop_diff_A[1:j-1])
                current_prop_A = np.sum(prop_diff_A[0:K-1]) + np.sum(prop_diff_A[1:j])
                
                # conditions to jump to the left
                if u >= previous_prop_A/prop_total_system and u < current_prop_A/prop_total_system:
                    
                    # adjust molecule numbers
                    molecules_A[j-1] -= 1
                    molecules_A[j-2] += 1
                    
                    # recompute propensities 
                    prop_diff_A[j-1] = molecules_A[j-1] * d_A                    
                    k_1[j-1] = molecules_A[j-1] * (molecules_A[j-1] - 1) * molecules_B[j-1] * k1
                    k_3[j-1] = molecules_A[j-1] * k3
                    
                    prop_diff_A[j-2] = molecules_A[j-2] * d_A
                    k_1[j-2] = molecules_A[j-2] * (molecules_A[j-2] - 1) * molecules_B[j-2] * k1
                    k_3[j-2] = molecules_A[j-2] * k3
                    
                    # break out of loop
                    break
        
        # compute if B_i jumps to the right
        elif u < (np.sum(prop_diff_A[0:(K-1)]) + 
                  np.sum(prop_diff_A[1:K]) + 
                  np.sum(prop_diff_B[0:(K-1)]))/prop_total_system:
            
            # find the compartment the jump happened in
            for j in range(1,K):
                
                previous_prop_B = (np.sum(prop_diff_A[0:K-1]) + 
                                   np.sum(prop_diff_A[1:K]) + 
                                   np.sum(prop_diff_B[:j-1]))
                
                current_prop_B = (np.sum(prop_diff_A[0:K-1]) + 
                                  np.sum(prop_diff_A[1:K]) + 
                                  np.sum(prop_diff_B[:j]))
                
                # conditions to jump to the right
                if u >= previous_prop_B/prop_total_system and u < current_prop_B/prop_total_system:
                    
                    # adjust molecule numbers
                    molecules_B[j-1] -= 1
                    molecules_B[j] += 1
                    
                    # recompute propensities
                    prop_diff_B[j-1] = molecules_B[j-1] * d_B
                    k_1[j-1] = molecules_A[j-1] * (molecules_A[j-1] - 1) * molecules_B[j-1] * k1
                    
                    prop_diff_B[j] = molecules_B[j] * d_B
                    k_1[j] = molecules_A[j] * (molecules_A[j] - 1) * molecules_B[j] * k1
                    
                    # break out of loop
                    break
            
        # compute if B_i jumps to the left
        elif u < (np.sum(prop_diff_A[0:(K-1)]) + 
                      np.sum(prop_diff_A[1:K]) + 
                      np.sum(prop_diff_B[0:(K-1)]) +
                      np.sum(prop_diff_B[1:K]))/prop_total_system:
            
            # find the compartment the jump takes place in 
            for j in range(2, K+1):
                previous_prop_B = (np.sum(prop_diff_A[0:K-1]) + 
                                  np.sum(prop_diff_A[1:K]) + 
                                  np.sum(prop_diff_B[0:K-1]) +
                                  np.sum(prop_diff_B[1:j-1]))
                
                current_prop_B = (np.sum(prop_diff_A[0:K-1]) + 
                                  np.sum(prop_diff_A[1:K]) + 
                                  np.sum(prop_diff_B[0:K-1]) +
                                  np.sum(prop_diff_B[1:j]))
                
                # conditions to jump to the left
                if u >= previous_prop_B/prop_total_system and u < current_prop_B/prop_total_system:
                    
                    # adjust molecule numbers
                    molecules_B[j-1] -= 1
                    molecules_B[j-2] += 1
                    
                    # recompute propensities 
                    prop_diff_B[j-1] = molecules_B[j-1] * d_B
                    k_1[j-1] = molecules_A[j-1] * (molecules_A[j-1] - 1) * molecules_B[j-1] * k1
                    
                    prop_diff_B[j-2] = molecules_B[j-2] * d_B
                    k_1[j-2] = molecules_A[j-2] * (molecules_A[j-2] - 1) * molecules_B[j-2] * k1
                    
                    # break out of loop
                    break
        
        # compute if 2A + B -> 3A
        elif u < (np.sum(prop_diff_A[0:(K-1)]) + 
                      np.sum(prop_diff_A[1:K]) + 
                      np.sum(prop_diff_B[0:(K-1)]) +
                      np.sum(prop_diff_B[1:K]) +
                      np.sum(k_1))/prop_total_system:
            
            # find the compartment reaction occurs in 
            for j in range(K+1):
                
                previous_prop = (np.sum(prop_diff_A[0:K-1]) + 
                                 np.sum(prop_diff_A[1:K]) + 
                                 np.sum(prop_diff_B[0:K-1]) +
                                 np.sum(prop_diff_B[1:K]) + 
                                 np.sum(k_1[0:j-1]))
                
                current_prop = (np.sum(prop_diff_A[0:K-1]) + 
                                np.sum(prop_diff_A[1:K]) + 
                                np.sum(prop_diff_B[0:K-1]) +
                                np.sum(prop_diff_B[1:K]) + 
                                np.sum(k_1[0:j]))
                
                if (u >= previous_prop/prop_total_system and 
                    u < current_prop/prop_total_system and
                    molecules_B[j-1] != 0):
                    
                    # B reduces by 1 and A increases by 1
                    molecules_B[j-1] -= 1
                    molecules_A[j-1] += 1
                    
                    # recompute the propensities
                    prop_diff_A[j-1] = molecules_A[j-1] * d_A
                    prop_diff_B[j-1] = molecules_B[j-1] * d_B
                    k_1[j-1] = molecules_A[j-1] * (molecules_A[j-1] - 1) * molecules_B[j-1] * k1
                    k_3[j-1] = molecules_A[j-1] * k3
                    
                    # break out of loop
                    break
        
        # compute if 0 -> A
        elif u < (np.sum(prop_diff_A[0:(K-1)]) + 
                      np.sum(prop_diff_A[1:K]) + 
                      np.sum(prop_diff_B[0:(K-1)]) +
                      np.sum(prop_diff_B[1:K]) +
                      np.sum(k_1) +
                      np.sum(k_2))/prop_total_system:
            
            # find the compartment reaction occured in
            for j in range(K+1):
                
                previous_prop = (np.sum(prop_diff_A[0:K-1]) + 
                                 np.sum(prop_diff_A[1:K]) + 
                                 np.sum(prop_diff_B[0:K-1]) +
                                 np.sum(prop_diff_B[1:K]) + 
                                 np.sum(k_1) +
                                 np.sum(k_2[0:j-1]))
                
                current_prop = (np.sum(prop_diff_A[0:K-1]) + 
                                 np.sum(prop_diff_A[1:K]) + 
                                 np.sum(prop_diff_B[0:K-1]) +
                                 np.sum(prop_diff_B[1:K]) + 
                                 np.sum(k_1) +
                                 np.sum(k_2[0:j]))
                
                if (u >= previous_prop/prop_total_system and u < current_prop/prop_total_system):
                    
                    # molecule of A is generated
                    molecules_A[j-1] += 1
                    
                    # recompute propensities
                    prop_diff_A[j-1] = molecules_A[j-1] * d_A
                    k_1[j-1] = molecules_A[j-1] * (molecules_A[j-1] - 1) * molecules_B[j-1] * k1
                    k_3[j-1] = molecules_A[j-1] * k3
                    
                    # break out of loop
                    break
        
        # compute if A -> 0
        elif u < (np.sum(prop_diff_A[0:(K-1)]) + 
                  np.sum(prop_diff_A[1:K]) + 
                  np.sum(prop_diff_B[0:(K-1)]) +
                  np.sum(prop_diff_B[1:K]) +
                  np.sum(k_1) +
                  np.sum(k_2) +
                  np.sum(k_3))/prop_total_system:
            
            # compute which compartment the reaction occured in
            for j in range(K+1):
            
                previous_prop = (np.sum(prop_diff_A[0:K-1]) + 
                                 np.sum(prop_diff_A[1:K]) + 
                                 np.sum(prop_diff_B[0:K-1]) +
                                 np.sum(prop_diff_B[1:K]) + 
                                 np.sum(k_1) +
                                 np.sum(k_2) + 
                                 np.sum(k_3[0:j-1]))
                
                current_prop = (np.sum(prop_diff_A[0:K-1]) + 
                                 np.sum(prop_diff_A[1:K]) + 
                                 np.sum(prop_diff_B[0:K-1]) +
                                 np.sum(prop_diff_B[1:K]) + 
                                 np.sum(k_1) +
                                 np.sum(k_2) + 
                                 np.sum(k_3[0:j]))
                
                if (u >= previous_prop/prop_total_system and 
                    u < current_prop/prop_total_system and
                    molecules_A[j-1] != 0):
                    
                    # a molecule of A is degraded
                    molecules_A[j-1] -= 1
                    
                    # recompute propensities
                    prop_diff_A[j-1] = molecules_A[j-1] * d_A
                    k_1[j-1] = molecules_A[j-1] * (molecules_A[j-1] - 1) * molecules_B[j-1] * k1
                    k_3[j-1] = molecules_A[j-1] * k3
                    
                    # break out of loop
                    break
        
        # compute if 0 -> B
        elif u < (np.sum(prop_diff_A[0:(K-1)]) + 
                  np.sum(prop_diff_A[1:K]) + 
                  np.sum(prop_diff_B[0:(K-1)]) +
                  np.sum(prop_diff_B[1:K]) +
                  np.sum(k_1) +
                  np.sum(k_2) +
                  np.sum(k_3) + 
                  np.sum(k_4))/prop_total_system:
            
            # find the compartment the reaction occured in
            for j in range(K+1):
                
                previous_prop = (np.sum(prop_diff_A[0:K-1]) + 
                                 np.sum(prop_diff_A[1:K]) + 
                                 np.sum(prop_diff_B[0:K-1]) +
                                 np.sum(prop_diff_B[1:K]) + 
                                 np.sum(k_1) +
                                 np.sum(k_2) + 
                                 np.sum(k_3) +
                                 np.sum(k_4[0:j-1]))
                
                current_prop = (np.sum(prop_diff_A[0:K-1]) + 
                                 np.sum(prop_diff_A[1:K]) + 
                                 np.sum(prop_diff_B[0:K-1]) +
                                 np.sum(prop_diff_B[1:K]) + 
                                 np.sum(k_1) +
                                 np.sum(k_2) + 
                                 np.sum(k_3) +
                                 np.sum(k_4[0:j]))
        
                if u >= previous_prop/prop_total_system and u < current_prop/prop_total_system:
                    
                    # a molecule of B is produced
                    molecules_B[j-1] += 1
                    
                    # recompute propensities
                    prop_diff_B[j-1] = molecules_B[j-1] * d_B
                    k_1[j-1] = molecules_A[j-1] * (molecules_A[j-1] - 1) * molecules_B[j-1] * k1
                    
                    # break out of the loop
                    break
    
        # compute new total propensity 
        prop_total_system = (np.sum(prop_diff_A[0:K-1]) + np.sum(prop_diff_A[1:K]) +
                             np.sum(prop_diff_B[0:K-1]) + np.sum(prop_diff_B[1:K]) +
                             np.sum(k_1) + np.sum(k_2) + np.sum(k_3) + np.sum(k_4))
        
    # create a histogram
    x_values = np.arange(1, K+1)
    
    # plot the number of A molecules in each compartment
    fig1 = plt.figure()
    plt.bar(x_values, molecules_A)
    plt.xlabel('Compartments')
    plt.ylabel('Number of Molecules A')
    plt.title('Q5) Schnakenberg A molecules')
    plt.show()
    
    # plot the number of B molecules in each compartment
    fig2 = plt.figure()
    plt.bar(x_values, molecules_B)
    plt.xlabel('Compartments')
    plt.ylabel('Number of Molecules B')
    plt.title('Q5) Schnakenberg B molecules')
    plt.show()
