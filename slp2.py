from random import choice
from numpy import array, dot, random
#from pylab import plot, ylim
import numpy as np
import matplotlib.pyplot as plt

unit_step = lambda x: 0 if x < 0 else 1
#print()

training_data = [
    (array([0,0,1]), 0),
    (array([0,1,1]), 0),
    (array([1,0,1]), 0),
    (array([1,1,1]), 1),
]

w = random.rand(3)
j=np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1],
])
errors = []
eta = 0.2
n = 100
for i in range(n):
    x, expected = choice(training_data)
    result = dot(w, x)
    error = expected - unit_step(result)
    errors.append(error)
    w += eta * error * x
  
#fig,ax = plt.subplots()
l=[]
for x, _ in training_data:
    result = dot(x, w)
    print("{}: {} -> {}".format(x[:2], result, unit_step(result)))
    l.append(unit_step(result))
    #ax.plot(x[0],(-w[0]*x[0])/w[1])
    #print(x[0],(-w[0]*x[0])/w[1])
    #ax.plot(x[:1], ((-w[:1]*x[:1])/w[2])
print(l)
for d, sample in enumerate(j):
    # Plot the negative samples
    print(sample[0],sample[1])
    if l[d] < 1:
        plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
    # Plot the positive samples
    else:
        plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)

plt.plot([0,1.2],[1.2,0])