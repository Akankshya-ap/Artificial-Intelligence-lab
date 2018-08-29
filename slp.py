# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 09:50:13 2018

@author: STUDENT1
"""

import matplotlib.pyplot as plt
import numpy as np
import random
# Weights (Synapses)
w = [0,0]

# Theshold
threshold = 0

# Bias
bias = 1

# Learning rate - [0, 1]
learning_rate = 1

# Maximum number allowed of iterations
max_iterations = 100
# Inputs and Labels
x = [
    [0.72,0.82,-1],
    [0.91,-0.69,-1],
    [0.03,0.93,-1],
    [0.12,0.25,-1],
    [0.96,0.47,-1],
    [0.8,-0.75,-1],
    [0.46,0.98,-1],
    [0.66,0.24,-1],
    [0.72,-0.15,-1],
    [0.35,0.01,-1],
    [-0.11,0.1,1],
    [0.31,-0.96,1],
    [0.0,-0.26,1],
    [-0.43,-0.65,1],
    [0.57,-0.97,1],
    [-0.72,-0.64,1],
    [-0.25,-0.43,1],
    [-0.12,-0.9,1],
    [-0.58,0.62,1],
    [-0.77,-0.76,1]
]
# Output
y = 0
# Color - Red or Blue, 1 and -1, respectively
color = ""
# Answer = Correct or Error
answer = ""


# Data Dictionary
data_dictionary = {
    # 'Keys' : 'Values',
    '0.72,0.82' : '-1',
    '0.91,-0.69' : '-1',
    '0.03,0.93' : '-1',
    '0.12,0.25' : '-1',
    '0.96,0.47' : '-1',
    '0.8,-0.75' : '-1',
    '0.46,0.98' : '-1',
    '0.66,0.24' : '-1',
    '0.72,-0.15' : '-1',
    '0.35,0.01' : '-1',
    '-0.11,0.1' : '1',
    '0.31,-0.96' : '1',
    '0.0,-0.26' : '1',
    '-0.43,-0.65' : '1',
    '0.57,-0.97' : '1',
    '-0.72,-0.64' : '1',
    '-0.25,-0.43' : '1',
    '-0.12,-0.9' : '1',
    '-0.58,0.62' : '1',
    '-0.77,-0.76' : '1'
}

def get_points_of_color(data, label):
    x_coords = [float(point.split(",")[0]) for point in data.keys() if data[point] == label]
    y_coords = [float(point.split(",")[1]) for point in data.keys() if data[point] == label]
    return x_coords, y_coords
# Turn on the interactive graphics mode
plt.ion()

for k in range(1, max_iterations):
    hits = 0
    print("\n------------------------- ITERATION "+str(k)+" ------------------------- ")

    for i in range(0,len(x)):
        sum = 0

        # Weighted sum
        for j in range(0,len(x[i])-1):
            sum += x[i][j] * w[j]

        # Output = Bias + Weighted sum
        output = bias + sum

        # Output is determined by the Threshold
        if output > threshold:
            y = 1
        else:
            y = -1     

        # Update the Weights if the output does not match with the Desired output
        if y == x[i][2]:
            hits += 1
            answer = "Correct!"
        else:
            for j in range (0,len(w)):             
                w[j] = w[j] + (learning_rate * x[i][2] * x[i][j])
            bias = bias + learning_rate * x[i][2]
            answer = "Error - Updating weight to: "+str(w)

        # Prints the answer
        if y == 1:
            print("\n"+answer)
        elif y == -1:
            print("\n"+answer)

        # Plot the graph
        plt.clf() # Clear figure
        plt.title('Iteration %s\n' % (str(k)))
        plt.grid(False) # Plot a grid
        plt.xlim(-1,1) # Set x-axis limits
        plt.ylim(-1,1) # Set y-axis limits

        xA = 1
        xB = -1

        if w[1] != 0:
            yA = (- w[0] * xA - bias) / w[1]
            yB = (- w[0] * xB - bias) / w[1]
        else:
            xA = - bias / w[0]
            xB = - bias / w[0]

            yA = 1
            yB = -1

        # Plot the black line, that is, we want to learn the black line as faithfully as possible
        #("Best" Decision Boundary)
        plt.plot([0.77, -0.55], [-1, 1], color='k', linestyle='-', linewidth=1)

        # Generates our green line, that is, our learning line (Decision Boundary)
        plt.plot([xA, xB], [yA, yB], color='g', linestyle='-', linewidth=2)

        # Plot blue points
        x_coords, y_coords = get_points_of_color(data_dictionary, '-1')
        plt.plot(x_coords, y_coords, 'bo')

        # Plot red points
        x_coords, y_coords = get_points_of_color(data_dictionary, '1')
        plt.plot(x_coords, y_coords, 'ro')

        # Highlights the current point
        #if answer == 'Correct!':
            # Correct - with green color
            #plt.plot(x[i][0], x[i][1], 'go', markersize=15, alpha=.5)
        #else:
            # Incorrect - with magenta color
            #plt.plot(x[i][0], x[i][1], 'mo', markersize=30, alpha=.5)

        # Show the plot
        plt.show()

        # We were able to control the loop time, so a figure will be updated and displayed
        plt.pause(0.05)

    if hits == len(x):
        print("\n---------------------------------------------------------------")
        print("\nFunctionality learned with "+str(k)+" iterations!")
        break;
    '''
    else:
        print("\n---------------------------------------------------------------")
        print("\nFunctionality not learned!")
        break;
    '''
    
print("\nDone!\n")