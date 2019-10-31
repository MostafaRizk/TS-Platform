from multiprocessing import Pool
from algorithm.parameters import params
from fitness.evaluation import evaluate_fitness
from stats.stats import stats, get_stats
from utilities.stats import trackers
from operators.initialisation import initialisation
from utilities.algorithm.initialise_run import pool_init

#For logging best rules
from os import path, getcwd, pardir
from utilities.algorithm.command_line_parser import parse_cmd_args
#import time

def search_loop():
    """
    This is a standard search process for an evolutionary algorithm. Loop over
    a given number of generations.
    
    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """

    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)

    # Initialise population
    individuals = initialisation(params['POPULATION_SIZE'])

    #t1 = time.time()
    # Evaluate initial population
    individuals = evaluate_fitness(individuals)
    #t2 = time.time()
    #print("Init time: "+str(t2-t1))

    # Generate statistics for run so far
    get_stats(individuals)

    phenoFileData = []
    dotsFileData = []

    # Traditional GE
    for generation in range(1, (params['GENERATIONS']+1)):
        stats['gen'] = generation
	
        #Save best phenotype and degree of task specialisation for every 5th generation
        if generation%5 == 0:
            best = max(individuals)
            
            #Save phenotype data
            phenoFileData += [best.phenotype]
            
            #Save DoTS data
            dotsFileData += [str(best.degree_of_task_specialisation)]
	
        # New generation
        #t3 = time.time()
        individuals = params['STEP'](individuals)
        #t4 = time.time()
        #print("Gen "+str(generation)+" time: "+str(t2-t1))

    #Log saved data about best phenotypes (i.e. rules)
    phenoFilename = path.join(getcwd(), pardir, pardir, 'argos-code/results/' + params['EXPERIMENT_NAME'] + "_" + str(params['GENERATIONS']) + '_' + str(params['POPULATION_SIZE']) + '_' + str(params['SLOPE']) + "_"  + str(params['RANDOM_SEED']) + '_all_rules' + '.txt')
    phenoFile = open(phenoFilename, 'a')
    for item in phenoFileData:
        phenoFile.write(item)
        phenoFile.write("\n")
    phenoFile.close()
    
    #Log saved data about DoTS
    dotsFilename = path.join(getcwd(), pardir, pardir, 'argos-code/results/' + params['EXPERIMENT_NAME'] + "_" + str(params['GENERATIONS']) + '_' + str(params['POPULATION_SIZE']) + '_' + str(params['SLOPE']) + "_"  + str(params['RANDOM_SEED']) + '_DoTS' + '.txt')
    dotsFile = open(dotsFilename, 'a')
    for item in dotsFileData:
        dotsFile.write(item)
        dotsFile.write("\n")
    dotsFile.close()
    
    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()

    return individuals


def search_loop_from_state():
    """
    Run the evolutionary search process from a loaded state. Pick up where
    it left off previously.

    :return: The final population after the evolutionary process has run for
    the specified number of generations.
    """
    
    individuals = trackers.state_individuals
    
    if params['MULTICORE']:
        # initialize pool once, if mutlicore is enabled
        params['POOL'] = Pool(processes=params['CORES'], initializer=pool_init,
                              initargs=(params,))  # , maxtasksperchild=1)
    
    # Traditional GE
    for generation in range(stats['gen'] + 1, (params['GENERATIONS'] + 1)):
        stats['gen'] = generation
        
        # New generation
        individuals = params['STEP'](individuals)
    
    if params['MULTICORE']:
        # Close the workers pool (otherwise they'll live on forever).
        params['POOL'].close()
    
    return individuals
