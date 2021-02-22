import ray
import time

ray.init()

@ray.remote
def f(x):
    time.sleep(20)
    return x

results = []
for i in range(2):
    results += [f.remote(i)]
    #results += [f(i)]

print(ray.get(results))
#print(results)