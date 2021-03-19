import os

from glob import glob

def get_seed_file(results_folder, parameter_list):
    """
    Get seed file that matches a particular parameter list

    @param results_folder:
    @param parameter_list:
    @return:
    """
    # Get pre and post seed params for model
    parameter_list = ["centralised_rwg"] + parameter_list[2:-7]

    # If individual reward, allow seeds that used any number of agents
    if parameter_list[3] == "individual":
        parameter_list[7] = '*'

    # Get list of all seed files with matching parameters
    seedfile_prefix = "_".join([str(param) for param in parameter_list])
    seedfile_extension = ".npy"
    possible_seedfiles = glob(f'{results_folder}/{seedfile_prefix}*{seedfile_extension}')

    # Return seed_file name but throw error if there's more than one
    if len(possible_seedfiles) == 0:
        raise RuntimeError('No valid seed files')
    elif len(possible_seedfiles) > 1:
        raise RuntimeError('Too many valid seed files')
    else:
        return possible_seedfiles[0]

#TODO: Adjust this for specialisation
def create_results_from_models(parent_folder, start_generation, step_size, num_generations):
    """
    Assemble results csv files from numpy files

    Currently only works for results_final.csv
    @param results_folder:
    @return:
    """
    results_folder = os.path.join(parent_folder, 'results')
    generation_list = [i for i in range(start_generation, num_generations+step_size, step_size)]
    generation_list += ["final"]

    for generation in generation_list:
        print(f"Doing results for generation {generation}")
        # Get list of all final models
        model_files = glob(f'{results_folder}/*cma*_{generation}.npy') #TODO: Allow different algorithms

        # Create final results file
        results_file = os.path.join(results_folder, f'results_{generation}.csv')
        f = open(results_file, 'w')

        # Write header
        header = "learning_type,algorithm_selected,team_type,reward_level,agent_type,environment,seed,num_agents,num_resources,sensor_range,sliding_speed,arena_length,arena_width,cache_start,slope_start,source_start,base_cost,upward_cost_factor,downward_cost_factor,carry_factor,resource_reward,episode_length,num_episodes, incremental_rewards,architecture,bias,hidden_layers,hidden_units_per_layer,activation_function,agent_population_size,sigma,generations,tolx,tolfunhist,seed_fitness,fitness,model_name"
        f.write(header)
        f.write("\n")

        # For every model, extract parameters and convert to a comma separated list
        for model_name in model_files:
            parameter_list = model_name.split("/")[-1].split("_")[0:-2]
            fitness = model_name.split("_")[-2]
            seed_file = get_seed_file(results_folder, parameter_list)
            seed_fitness = seed_file.split("_")[-1].strip(".npy")

            # Log parameters and score to results file
            parameters_to_log = parameter_list + [seed_fitness] + [fitness] + [model_name]
            line_to_log = ",".join(parameters_to_log)
            f.write(line_to_log)
            f.write(("\n"))

        # Close results file
        f.close()

create_results_from_models("/Users/mostafa/Documents/Code/PhD/TS-Platform/results/2021_03_16_equal_games_per_learner_sequential", start_generation=50, step_size=50, num_generations=1000)