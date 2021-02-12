import sys
import math
#simple gradient descent
#over j(b) = 1/2(b^2) + 7sin(b) + 8

if len(sys.argv) < 3:
    print("Please pass the initial beta as 1st argument, with learning rate as the 2nd argument...")
    exit()

bOrig = float(sys.argv[1])
lRate = float(sys.argv[2])

cost = bOrig + (7 * math.cos(bOrig)) 
b = bOrig - (lRate * (cost))
bPrev = bOrig
iter = 0

while b != bPrev:
    cost = b + (7 * math.cos(bOrig))
    bPrev = b
    b = bPrev - (lRate * cost)
    jb = (0.5 * (b ** 2)) + (7 * math.sin(b)) + 8
    if iter % 100 == 0:
        print(f"beta = {b}, J(beta) = {jb}, epochs = {iter}")
    iter = iter + 1
    
jb = (0.5 * (b ** 2)) + (7 * math.sin(b)) + 8

print(f"beta = {b}, J(beta) = {jb}, epochs = {iter}")