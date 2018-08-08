# -*- coding: utf-8 -*-
"""
Created on Wed Aug  8 11:09:33 2018

@author: STUDENT1
"""

def pour(jug1, jug2):
    max1, max2, fill = 5, 4,3  #Change maximum capacity and final capacity
    print("%d\t%d" % (jug1, jug2))
    if jug1 is fill:
        return
    elif jug1==0:
        pour(max1,jug2)
    elif jug1==max1 and jug2<max2:
        pour(jug1-(max2-jug2),max2)
    elif jug2>0:
        pour(jug1,0)
    elif jug1+jug2<=max2 and jug1>=0:
        pour(0,jug1+jug2)
print("JUG1\tJUG2")
pour(0, 0)
print("\nGoal is achieved")