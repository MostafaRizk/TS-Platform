import numpy as np

rng = np.random.RandomState(1)

for i in range(5):
    print(rng.randint(low=0, high=1000000))

print()
rng = np.random.RandomState(1)

for j in range(5):
    print(rng.randint(low=0, high=1000000))