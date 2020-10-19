from glob import glob
import re
import pandas as pd
from shutil import copyfile
import matplotlib.pyplot as plt


# Get all parameter files
def get_all_files():
    return [name.replace('data/experiments/', '') for name in glob(f'data/experiments/cma*json')]


# Find parameter files that weren't run
def get_unstarted_files():
    f = open('data/cma.out')
    output_file_data = f.read()
    return re.findall(r'cma.*json', output_file_data)


# Get number of seg faults
def count_seg_faults():
    f = open('data/cma.out')
    output_file_data = f.read()
    return re.findall(r'Segmentation fault', output_file_data)


# Get number of timeouts
def count_timeouts():
    f = open('data/cma.out')
    output_file_data = f.read()
    return re.findall(r'DUE TO TIME LIMIT', output_file_data)


# Find parameter files that seg-faulted, ran out of time or didn't run at all
def get_rerun_files(all_parameter_files):
    rerun_parameter_files = []
    data = pd.read_csv('data/results/results_final.csv')
    complete_parameter_files = []

    for index, row in data.iterrows():
        complete_parameter_files += ['_'.join(row["model_name"].split('_')[0:-3]) + '.json']

    for parameter_file in all_parameter_files:
        if parameter_file not in complete_parameter_files and parameter_file not in rerun_parameter_files:
            rerun_parameter_files += [parameter_file]

    return rerun_parameter_files


# Copy all the files that need re-running to a new folder and add them to LIST_cma
def copy_rerun_files(rerun_parameter_files):
    g = open(f"../LIST_cma", "a")

    for parameter_file in rerun_parameter_files:
        src = 'data/experiments/' + parameter_file
        dst = 'data/new_experiments/' + parameter_file
        copyfile(src, dst)
        g.write(f"python3 experiment.py --parameters {parameter_file}\n")

    g.close()


# Find files that started but didn't finish (seg-fault or ran out of time)
def get_incomplete_files(unstarted_parameter_files, rerun_parameter_files):
    incomplete_files = []

    for parameter_file in rerun_parameter_files:
        if parameter_file not in unstarted_parameter_files:
            incomplete_files += [parameter_file]

    return incomplete_files


def get_generation_count(incomplete_files):
    final_gen = {}
    y_vals = {}
    y_err_vals = {}

    for parameter_file in incomplete_files:
        final_gen[parameter_file] = -1
        y_vals[parameter_file] = []
        y_err_vals[parameter_file] = []

    for i in range(20, 1020, 20):
        results_file = f"data/results/results_{i}.csv"
        data = pd.read_csv(results_file)

        for index, row in data.iterrows():
            parameter_file = '_'.join(row["model_name"].split('_')[0:-3]) + '.json'

            if parameter_file in final_gen:
                final_gen[parameter_file] = i
                fitness = row["fitness"]
                y_vals[parameter_file] += [fitness]
                y_err_vals[parameter_file] += [0]


    return final_gen, y_vals, y_err_vals


all_parameter_files = get_all_files()
unstarted_parameter_files = get_unstarted_files()
rerun_parameter_files = get_rerun_files(all_parameter_files)
incomplete_files = get_incomplete_files(unstarted_parameter_files, rerun_parameter_files)
#copy_rerun_files(rerun_parameter_files)
#final_gen, y_vals, y_err_vals = get_generation_count(incomplete_files)

'''
# Plot
fig1, ax1 = plt.subplots(figsize=(12, 4))
ax1.set_title('Fitness Throughout Evolution')
ax1.set_ylim(0, 300000)
ax1.set_ylabel('Fitness')
ax1.set_xlabel('Generation')

x = [i for i in range(20, 1020, 20)]

for key in y_vals:
    plt.errorbar(x, y_vals[key], y_err_vals[key])

plt.savefig("evolution_history.png")
'''

#for key in final_gen:
#    print(final_gen[key])

num_segfaults = len(count_seg_faults())
num_timeouts = len(count_timeouts())
num_unstarted = len(unstarted_parameter_files)
num_reruns = len(rerun_parameter_files)

print(num_segfaults)
print(num_timeouts)
print(num_unstarted)
print(f'{num_segfaults + num_timeouts + num_unstarted} = {num_reruns}')