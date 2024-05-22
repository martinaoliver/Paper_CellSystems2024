
'''
====================================================
    Comments: Parallelisation.py is a script to run a function in parallel across different threads.
    The number of threads is set by the user.
    The function to be run in parallel is defined in the my_function() function.
    The function takes two arguments, b and c, and returns the sum of the two.


    This script should be runned in the HPC cluster for parallelisation. 
    To call this script, we use the parallelised_job.pbs script which calls:
    python parallelised_job.py Number_of_Threads
    where Number_of_Threads is an integer argument to the parallelised_job.py function.

    In this script we parallelise a computation which in this case is a simple sum of two numbers.
    We have a dataframe with many rows and want to sum the values of the two columns in parallel.
    This operation is not very computationally expensive, but imagine the function was more expensive and it had to be split in batches. 
====================================================
'''



'''
====================================================
    Imports
====================================================
'''
import sys
import multiprocessing
import pandas as pd
import numpy as np
import pickle as pkl
'''
====================================================
    Code
====================================================
'''

# Set number of threads to 1 if no valid number provided
if len(sys.argv) > 1:
    Number_of_Threads = int(sys.argv[1])
else:
    Number_of_Threads = 1
print('Number of Threads set to ', Number_of_Threads)

# Define function to run in parallel
def my_function(df_batch, path, filename, batch_count):
    df_batch["sum a+b"] = df_batch["a"] + df_batch["b"] #sums over all rows
    with open(path + f'{filename}{batch_count}.pkl', 'wb') as f: #saves the dataframe to a pickle file
        pkl.dump(df_batch, f)
    print(df)


# Create dataframe for input
df = pd.DataFrame(np.random.normal(size=(1000,2)), columns=['a', 'b'])
print(df)
print(f'Length of original dataframe = {len(df)}')

# Define batch size
batch_size = int(len(df)/Number_of_Threads) + 1
batch_indices = list(range(0, len(df), batch_size))
print(f'Length of batched dataframes = {batch_size}')

# Create a pool of workers
pool = multiprocessing.Pool(Number_of_Threads)

# Run my_function in parallel across different threads
pool_output = []
path = ''
filename = 'output_file_'
for batch_count, start_batch_index in enumerate(batch_indices):
    print('main' + str(start_batch_index))
    df_batch = df.iloc[start_batch_index:start_batch_index+batch_size] #splits the dataframe into batches
    pool_output.append(pool.apply_async(my_function, args=(df_batch, path, filename, batch_count))) #runs the function in parallel: sends a batch of the dataframe to the function my_function


# Close the parallel processing job
pool.close()
pool.join()




