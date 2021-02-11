import sys
import math
import numpy as np
#simple gradient descent
#over j(b) = 1/2(b^2) + 7sin(b) + 8

if len(sys.argv) < 3:
    print("Please pass in file as argument, with Y in the first column and x in 2nd columns delimited by commas. And pass learning rate as second argument")
    exit()
data = np.genfromtxt(sys.argv[1], delimiter=",")

y = data[:,0]
x = data[:,1]
lrate = float(sys.argv[2])
B0 = 0
B1 = 0
oldB0 = B0
oldB1 = B1

def pred(b0, x, b1):
    return b0 + x * b1

def calcGradients(B0, B1, x, y):
    b0_gradient = 0
    b1_gradient = 0
    n = len(y)
    for i in range(n):
        predictedY = pred(B0, x[i], B1)
        b0_gradient += predictedY - y[i]
        b1_gradient += (predictedY - y[i]) * x[i]
    B0 = B0 - lrate * b0_gradient * 1/n
    B1 = B1 - lrate * b1_gradient * 1/n
    #print(f"beta0 = {B0}, beta1 = {B1}")
    return [B0,B1]

def calcQ(B0, B1, x, y):
    q = 0
    n = len(y)
    for i in range(n):
        q += ((y[i] - (B0 + B1 * x[i])) ** 2)
    return q

#calcGradients gets plopped into B0 and B1
iter = 0
Q = calcQ(B0, B1, x, y)
oldQ = Q

while oldQ >= Q:
    bArr = calcGradients(B0, B1, x, y)
    oldB0 = B0
    oldB1 = B1
    B0 = bArr[0]
    B1 = bArr[1]
    oldQ = Q
    Q = calcQ(B0, B1, x, y)
    #if iter % 50 == 0:
    #    print(f"beta0 = {B0}, beta1 = {B1}, epochs = {iter}")
    iter += 1

iter -= 1
B0 = oldB0
B1 = oldB1

print(f"beta0 = {B0}, beta1 = {B1}, epochs = {iter}")