import numpy as np
import math
import sys

def squareArr(arr):
    return arr ** 2

def multArr(arr,arr2):
    return arr * arr2

#E[x] = average of x
#varianc(x) = E[(x - E[x])^2] = E[x^2] - E[x]^2
#covariance(x,y) = 𝐸[(𝑋−𝐸[𝑋])(𝑌−𝐸[𝑌])]=𝐸[𝑋𝑌]−𝐸[𝑋]𝐸[𝑌]
#cov matrix
#[cov(x,x)  cov(x,y)]
#[cov(y,x)  cov(y,y)]
if len(sys.argv) < 2:
    print("Please pass in file as argument, with Y in the first column and x in the second column delimited by commas...")
    exit()

data = np.genfromtxt(sys.argv[1], delimiter=",")

ex = np.average(data[:,0])
ey = np.average(data[:,1])
ex_2 = ex ** 2
ey_2 = ey ** 2
ex2 = np.average(np.array(list(map(squareArr,data[:,0]))))
ey2 = np.average(np.array(list(map(squareArr,data[:,1]))))
xy = np.array(list(map(multArr,data[:,0],data[:,1])))
exy = np.average(xy)
exey = ex * ey


varx = ex2 - ex_2
vary = ey2 - ey_2
covxy = exy - exey 

A = np.array([[float(varx),float(covxy)],[float(covxy),float(vary)]])

#Characteristic Equation Section
a = 1
b = (float(A[0,0]) + float(A[1,1])) * -1
c = (float(A[0,0]) * float(A[1,1])) - (float(A[0,1]) * float(A[1,0]))

# function for finding roots
def calcLamda(a, b, c):  
    lamda = np.array([float(0),float(0)])
    disc = (b * b) - (4 * a * c)

    if disc < 0:
        print("Complex root, how do you do eigenvalues?")
        return False
    else:
        sqrt_val = math.sqrt(disc)
        lamda[0] = (-b + sqrt_val) / (2 * a)
        lamda[1] = (-b - sqrt_val) / (2 * a) 
        if(lamda[0] < lamda[1]):
            l = lamda[0]
            lamda[0] = lamda[1]
            lamda[1] = l
        return lamda


lamda = calcLamda(a,b,c)

if type(lamda) == bool:
    exit()

#EigenVectors Section
def calculateEigenVectors(lamda):
    #eigenvectors
    U = np.array([[float(0),float(0)],[float(0),float(0)]])
    U[0,0] = float(1)
    U[1,0] = (lamda[0] - A[0,0]) / A[0,1]#1st equation here
    U[0,1] = float(1)
    U[1,1] = A[1,0] / (lamda[1] - A[1,1])#and for fun 2nd equation here

    H = np.array([math.sqrt((U[0,0] ** 2) + (U[1,0] ** 2)),math.sqrt((U[0,1] ** 2) + (U[1,1] ** 2))])
    U[0,0] /= H[0]
    U[1,0] /= H[0]
    U[0,1] /= H[1]
    U[1,1] /= H[1]

    return U

U = calculateEigenVectors(lamda)

print(f"Lamda1 = {lamda[0]}, and u1 = {U[:,0]}")
print(f"Lamda2 = {lamda[1]}, and u2 = {U[:,1]}")

#Symmetric section
def computeVDVT(V, Vt, D):
    ApTest = np.dot(np.dot(V,D),Vt)
    return ApTest

def calcVDVT(U, lamda, A):
    V = U
    D = np.array([[lamda[0],float(0)],[float(0),lamda[1]]])
    Vt = np.transpose(V)
    Ap = computeVDVT(V, Vt, D)

    if (math.isclose(A[0,0], Ap[0,0])
    & math.isclose(A[0,1], Ap[0,1])
    & math.isclose(A[1,0], Ap[1,0])
    & math.isclose(A[1,1], Ap[1,1])):
        print(f"V: {V}\nD: {D}\nVt: {Vt}")
        return True
    else:
        return False

def demensionalityReduction(l):
    if((l[0] / 10) >= l[1]):
        print("You might want to consider dimensionality reduction...")
    else:
        print("It looks like both variables are useful")

if calcVDVT(U, lamda, A):
    demensionalityReduction(lamda)
    exit()
elif calcVDVT(np.array([[U[0,1],U[0,0]],[U[1,1],U[1,0]]]), lamda, A):
    demensionalityReduction(lamda)
    exit()
else:
    print("Somthing went wrong computing VDVt")