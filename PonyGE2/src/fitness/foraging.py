from algorithm.parameters import params
from fitness.base_ff_classes.base_ff import base_ff
from os import path, getcwd, pardir, chdir
import subprocess


class foraging(base_ff):
    """Calculates fitness according to the function defined in the ARGoS Loop Functions"""

    maximise = True

    def __init__(self):
        # Initialise base fitness function class.
        super().__init__()

    def escape_characters(self, escape_string):
        escape_string = escape_string.replace(";","\;")
        return escape_string

    """
    def evaluate(self, ind, **kwargs):
        ruleFilename = path.join(getcwd(),pardir,pardir,'argos-code/foraging_rules.txt')
        rulefile = open(ruleFilename, 'w')
        rulefile.write(ind.phenotype)
        rulefile.close()
        #Call ARGoS here
        originalDir = getcwd()
        chdir('../../argos-code/')
        process = subprocess.Popen(['argos3 -c experiments/ge_foraging-training.argos'], shell=True)
        process.wait()
        chdir(originalDir)
        performanceFileName = path.join(getcwd(),pardir,pardir,'argos-code/foraging_performance.txt')
        performanceFile = open(performanceFileName,'r')
        fitness = int(performanceFile.read())
        return fitness
    """

    def evaluate(self, ind, **kwargs):
        originalDir = getcwd()
        chdir('../../argos-code/')
        process = subprocess.Popen(['build/embedding/ge_foraging/ge_foraging ' + self.escape_characters(ind.phenotype)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        #build/embedding/ge_foraging/ge_foraging P_ON_SOURCE,true\;P_HAS_OBJECT,false\;*:*:IS_WANT_OBJECT,true,68#
        output = process.communicate()
        output_string = str(output[0])
        error_string = str(output[1])
        #for i in output_string.split('\\n'):
        #    print(i)
        #print(output_string)
        #print("---")
        #print(error_string)
        #print("---")
        start_index = output_string.find("PERFORMANCE")
        end_index = output_string.find("PERFORMANCE", start_index+1)
        chdir(originalDir)
        fitness = float(output_string[start_index+len("PERFORMANCE")+1 : end_index-1])
        return fitness