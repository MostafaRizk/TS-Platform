This directory contains instructions for how to reproduce all the experiments in my thesis. Each experiment has its own sub-directory with the following folders:

    experiments- All the relevant files for running the experiments (.json, LIST and .job)
    results- The results from when I ran it
    analysis- The plots I use 

I ran my experiments on a university cluster using SLURM and a python virtual environment. If you intend on using a similar setup, below are some instructions to help.

Setting up a virtual environment:

1. Go to the directory that has the python files you’re running. 

2. Type in:

        module load python/3.6.2-static

If you’re using a different version of python, load that one instead. 

To view all the available python versions on your cluster, you can type:

       module avail
 
to see all modules available on the cluster and scroll to the python modules or you can type: 

    module load python 

and press tab twice to list just the relevant ones.

3. Type: 

        python3 -m venv env

This creates the virtual environment

4. Type:
   
        source env/bin/activate
    
This activates the virtual environment
You will also need to include this in your job file as well
Put the line 

    source path_to_env_folder/bin/activate 
 
 before the line that runs your experiments.

5. Install any python libraries that the project uses

        pip3 install [name of library]
