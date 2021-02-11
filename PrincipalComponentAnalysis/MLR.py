import numpy as np
#import math
import sys

if len(sys.argv) < 2:
    print("Please pass in file as argument, with Y in the first column and additional columns delimited by commas...")
    exit()
data = np.genfromtxt(sys.argv[1], delimiter=",")

length = data.shape[0]
x = np.array(data)
y = data[:,0]
i = 0
while i < length:
    x[i,0] = 1
    i += 1

xt = np.transpose(x)
xtx = np.dot(xt,x)
xty = np.dot(xt,y)

print("Y=\n",y)
print("X=\n", x)
print("X-Transpose X=\n", xtx)
print("X-Transpose Y=\n", xty)

if(np.linalg.det(xtx) != 0):
    inv = np.linalg.inv(xtx)
    print("Inverse=", inv)
else:
    inv = np.linalg.pinv(xtx)
    print("Psuedo-Inverse=\n", inv)

b = np.dot(inv,xty)
print("b=\n",b)