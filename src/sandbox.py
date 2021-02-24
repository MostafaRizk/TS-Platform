import ray
import time

from copy import deepcopy


@ray.remote
def parallel_learning(representatives, generations, index):
    other = [1, 0]

    print(f"Scores at generation {0}: {representatives[0]}")

    for generation in range(1, generations+1):
        time.sleep(1)
        other_teammate_index = other[index]

        while not representatives[generation-1][other_teammate_index]:
            #print(f"Agent {index} is waiting at generation {generation}")
            continue

        if index % 2 == 0:
            representatives[generation][index] = representatives[generation-1][index] + 2
        else:
            representatives[generation][index] = representatives[generation-1][index] + 4

        print(f"Scores at generation {generation}: {representatives[generation]}")

    return representatives[index]

@ray.remote
def parallel_learning2(representatives, index):
    time.sleep(1)

    if index % 2 == 0:
        return representatives[index] + 2
    else:
        return representatives[index] + 4


class DummyLearner:
    def __init__(self):
        self.generations = 100
        self.num_agents = 2
        self.representatives = [[1] * self.num_agents]

        for g in range(1, self.generations+1):
            self.representatives += [[None] * self.num_agents]

    def learn(self):
        ray.init(num_cpus=self.num_agents)
        representatives = ray.put(self.representatives)

        parallel_threads = []

        for agent in range(self.num_agents):
            parallel_threads += [parallel_learning.remote(representatives, self.generations, agent)]

        return ray.get(parallel_threads)


class DummyLearner2:
    def __init__(self):
        self.generations = 100
        self.num_agents = 2
        self.representatives = [1] * self.num_agents

    def learn(self):
        ray.init(num_cpus=self.num_agents)

        for generation in range(self.generations):
            parallel_threads = []

            for agent in range(self.num_agents):
                parallel_threads += [parallel_learning2.remote(self.representatives, agent)]

            self.representatives = ray.get(parallel_threads)

        return self.representatives


class DummyLearner3:
    def __init__(self):
        self.generations = 100
        self.num_agents = 2
        self.representatives = [1] * self.num_agents

    def learn(self):
        for generation in range(self.generations):
            for agent in range(self.num_agents):
                time.sleep(1)

                if agent % 2 == 0:
                    self.representatives[agent] += 2
                else:
                    self.representatives[agent] += 4

        return self.representatives


parallel_learner = DummyLearner()
#parallel_learner = DummyLearner2()
results = parallel_learner.learn()
#serial_learner = DummyLearner3()
#results = serial_learner.learn()
print(results)


