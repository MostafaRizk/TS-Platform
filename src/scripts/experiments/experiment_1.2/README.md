1. Run the commands in experiments/LIST_rwg. If you are using SLURM, you can modify the provided job file and run it on your cluster using:

        sbatch rwg.job

2. Run:

        ./make_plots.sh PATH_TO_YOUR_RESULTS_DIRECTORY
   
   where PATH_TO_YOUR_RESULTS_DIRECTORY is the directory containing all the results of step 1.