This is a tutorial to access and send jobs to the High Performance Computing cluster at Imperial College London. 

# HPC Cluster access

Permission must be requested to your supervisor or directly to the HPC team https://www.imperial.ac.uk/admin-services/ict/self-service/research-support/rcs/.

Once permission is given, access through the terminal:

```bash
$ ssh username@login.hpc.ic.ac.uk
```

If connected to a non-imperial wifi network, make sure the imperial VPN is on. 

# Send jobs to the HPC

Jobs are not sent directly in the terminal as we would do in our own terminal

```bash
$ python job.py
```


Instead, they are sent through .pbs files, which determine the number of cpus, walltime and memory needed. 

An example of a .pbs file to run in python parallelised_job.py is shown below. The file is called parallelised_job.pbs 


```bash
#!/bin/sh
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=10:mem=96gb



module load anaconda3/personal
source activate env1
cd $PBS_O_WORKDIR


python parallelised_job.py 10


```

In this case, we have requested 20 minutes, 10 cpus and 96gb memory to run parallelised_job.py. The higher the walltime, ncpus or mem, the longer the queue for your job to start. 



Once the .pbs file is created, we can send it to the cluster where a job with ID x gets queued to run

```bash
$ qsub parallelised_job.pbs
x.pbs
```
To delete the job request, 
```bash
$ qdel x.pbs
```

To get in information on the queue status 

```bash
$ qstat

or 

$ qstat -w -T

```

The parallelised_job.pbs file sends parallelised_job.py to run in the cluster cores. The terminal output and the errors of job x are stored in 
<em>parallelised_job.pbs.ox</em> and <em>parallelised_job.pbs.ex</em>.
They can be visualized with nano or vim in the terminal.

## Virtual environments

To activate a virtual environment (env) in the cluster, use the commands
```
$ module load anaconda3/personal
$ source activate env
```

To install python packages, run in the terminal 

```
$ pip install package 
or 
$conda install package
```

The .pbs file should also have a command to activate the virtual environment if packages are needed.


```bash

#!/bin/sh
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=10:mem=96gb

$ module load anaconda3/personal
$ source activate env
cd $PBS_O_WORKDIR


python parallelised_job.py 10

```

## Command line arguments

If a command must be passed to the .pbs file, this is included in the terminal command as 

```
$ qsub -v arg1=x,arg2=y parallelised_job.pbs
```

where parallelised_job.pbs is


```bash

#!/bin/sh
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=10:mem=96gb

$ module load anaconda3/personal
$ source activate env
cd $PBS_O_WORKDIR


python parallelised_job.py 10 $arg1 $arg2

```

