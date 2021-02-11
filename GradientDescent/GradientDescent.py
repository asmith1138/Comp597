import sys
import math
#simple gradient descent
#over j(b) = 1/2(b^2) + 7sin(b) + 8

if len(sys.argv) < 3:
    print("Please pass the initial beta as 1st argument, with learning rate as the 2nd argument...")
    exit()

bOrig = float(sys.argv[1])
lRate = float(sys.argv[2])

jbOrig = (0.5 * (bOrig ** 2)) + (7 * math.sin(bOrig)) + 8
jb = jbOrig
jbPrev = jbOrig
b = bOrig
bPrev = bOrig
n = 0

while jb <= jbPrev:
    jbPrev=jb
    bPrev = b
    b = b + lRate
    jb = (0.5 * (b ** 2)) + (7 * math.sin(b)) + 8
    n = n + 1
    #print(f"b={b},jb={jb},n={n}")

if n == 1:
    jb = jbPrev
    b = bPrev
    n = n - 1
    while jb <= jbPrev:
        jbPrev=jb
        bPrev = b
        b = b - lRate
        jb = (0.5 * (b ** 2)) + (7 * math.sin(b)) + 8
        n = n + 1
        #print(f"b={b},jb={jb},n={n}")

jb = jbPrev
b = bPrev
n = n - 1

print(f"beta = {b}, J(beta) = {jb}, epochs = {n}")