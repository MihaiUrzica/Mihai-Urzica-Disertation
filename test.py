# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 17:24:35 2021

@author: Mihai
"""

S=['c','d','e','o']
A=[3,2,0,1]
n=4
f=''
for i in range(0,n-2) :
    f = f + S[i]
    f = f + S[A[i]]
    if A[i] == 0:
        break
print(f)