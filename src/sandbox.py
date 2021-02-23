import ray
import time

ray.init()

@ray.remote
def f(x):
    time.sleep(1)
    return x, x*10, x*100

@ray.remote
def empty():
    return None

results = []
for i in range(2):
    results += [f.remote(i)]
    #results += [f(i)]
results += [empty.remote()]

print(ray.get(results))
#print(results)