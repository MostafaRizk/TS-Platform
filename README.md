# TS-Platform
This is an experimental platform for studying the challenges of learning cooperation in a Multi-Agent System. The code consists of 3 main modules:

## src/agents ## 
Implementations of agent controllers. This includes NN-based agents capable of learning policies and hardcoded agents for benchmarking.

## src/envs ## 
The main environment in this platform is the Slope Foraging task implemented in slope.py. This is a task where agents must retrieve resources from the top of a slope and benefit from cooperating. The experimental platform is highly modular, so other environments can be implemented and plugged in. The same is true for agent controllers and learning algorithms.

## src/learning ## 
Implementations of centralised and decentralised learning algorithms.

## Running Experiments ##
To conduct experiments, run 

python3 experiment.py —parameters parameter_file.json

where parameter_file.json is a file containing the experiment configurations. Example json files can be found in src.

## Publications arising from this implementation ##

M. Rizk, J. Garcia, A. Aleti, D. Green, “Using Evolutionary Game Theory to Understand Scalability in Task Allocation” Poster accepted at: The Genetic and Evolutionary Computation Conference (GECCO); 2022 Jul 09-13; Boston, MA.
