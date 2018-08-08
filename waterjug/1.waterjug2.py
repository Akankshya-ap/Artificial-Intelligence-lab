# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 10:37:57 2018

@author: STUDENT1
"""

import fractions
import math
class WaterJug:
        
        def __init__(self, x, y, goal):
            self.JUG_1 = x
            self.JUG_2 = y
            self.j_1 = 0
            self.j_2 = 0
            self.pre_j_1 = 0
            self.pre_j_2 = 0
            self.goal = goal
            self.path = []
                              
        def get_path(self):                  
            pass
            
        def next_state(self, next_act):
            self.path.append(self.j_1, self.j_2)
            if self.j_1 < self.JUG_1:
                next_act.append(self.JUG_1, self.j_2)
                
            if self.j_2 < self.JUG_2:
                next_act.append(self.j_1, self.JUG_2)
                
            if self.j_1 > 0:
                next_act.append(0, self.j_2)
                
            if self.j_2 > 0:
                next_act.append(self.j_1, 0)
                
            if self.j_1 + self.j_2 >= self.JUG_1 and self.j_2 > 0:
                next_act.append(self.JUG_1, self.j_2 - (self.JUG_1 - self.j_1))
                
            if self.j_1 + self.j_2 >= 0 and self.j_1 > 0:
                next_act.append(self.j_1- (self.JUG_2 - self.j_2), self.JUG_2)
                
            if self.j_1 + self.j_2 <= self.JUG_1 and self.j_2 > 0:
                next_act.append(self.j_1 + self.j_2, 0)
                
            if self.j_1 + self.j_2 <= self.JUG_2 and self.j_1 > 0:
                next_act.append(0, self.j_1 + self.j_2)
        
if __name__ == "__main__":
    x = int(input( "Enter the capacity of first Water jug : "))
    y = int(input("Enter the Capacity of the second water jug : "))
    goal = int(input("Enter the amount you want in the first jug : "))
    problem = WaterJug(x, y, goal)
    hcf = math.gcd( problem.JUG_1, problem.JUG_2)
    if goal % hcf == 0:
        problem.next_state()
    else: 
        print ("No Solution exists")