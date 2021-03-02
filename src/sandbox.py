import ray

class FitnessDummy:
    def __init__(self):
        pass

    def calc_fit(self, x):
        return x

@ray.remote
def learn_in_parallel(fitness_calc, x):
    return fitness_calc.calc_fit(x)

class LearnerDummy:
    def __init__(self):
        self.fitness_calc = FitnessDummy()

    def learn(self):
        ray.init(num_cpus=2)
        parallel_threads = []

        for i in range(2):
            parallel_threads += [learn_in_parallel.remote(self.fitness_calc, i)]

        parallel_results = ray.get(parallel_threads)

        return parallel_results

dummy = LearnerDummy()
results = dummy.learn()
print(results)



