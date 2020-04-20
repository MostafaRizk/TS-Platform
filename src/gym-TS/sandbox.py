import cma
import numpy as np

es = cma.CMAEvolutionStrategy([1,2,3,4], 0.1)

solutions = es.ask()
new_solutions = solutions

print(solutions)
new_solutions = np.append(new_solutions, solutions, axis=0)
print(new_solutions)

es.tell(new_solutions, [1,2,3,4,5,6,7,8,1,2,3,4,5,6,7,8])