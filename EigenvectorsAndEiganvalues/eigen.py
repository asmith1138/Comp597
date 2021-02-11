import numpy as np
import math

#Input Section
def takeinputs():
    a11 = input("Enter Matrix value 1,1: ") 
    a12 = input("Enter Matrix value 1,2: ") 
    a21 = input("Enter Matrix value 2,1: ") 
    a22 = input("Enter Matrix value 2,2: ") 
    return np.array([[float(a11),float(a12)],[float(a21),float(a22)]])

A = takeinputs()

#Characteristic Equation Section
#Au = lamda u
#A - lamda I is linearly dependent
#a - lamda b
#c         d - lamda
#det = 0 is ad - bc = 0
#lamda^2 + (a11+a22)*-lamda + (a11a22 - a12a21) = 
#a11d22 - (a11+a22)lamda + lamda^2 - a12a21 = 0
#a = int(A[0][0]) * int(A[1][1])
#b = int(A[0][0]) + int(A[1][1]) + 1
#c = int(A[1][0]) * int(A[0][1])
a = 1
b = (int(A[0,0]) + int(A[1,1])) * -1
c = (int(A[0,0]) * int(A[1,1])) - (int(A[0,1]) * int(A[1,0]))

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
        return lamda


lamda = calcLamda(a,b,c)

if type(lamda) == bool:
    exit()

#EigenVectors Section
def calculateEigenVectors(lamda):
    #eigenvectors
    #a11*u1 + a12*u2 = lamda*u1
    #a21*u1 + a22*u2 = lamda*u2
    #solve for u1 and u2
    #for each lamda
    #(lamda-a11)u1 / a12 = u2
    #(lamda-a22)u2 / a21 = u1
    #make u1 = 1
    U = np.array([[float(0),float(0)],[float(0),float(0)]])
    U[0,0] = float(1)
    U[1,0] = (lamda[0] - A[0,0]) / A[0,1]#1st equation here
    U[0,1] = float(1)
    U[1,1] = A[1,0] / (lamda[1] - A[1,1])#and for fun 2nd equation here

    #u1^2 + u2^2 = h^2
    # h = sqrt(u1^2 + u2^2)
    #u1/h and u2/2
    H = np.array([math.sqrt((U[0,0] ** 2) + (U[1,0] ** 2)),math.sqrt((U[0,1] ** 2) + (U[1,1] ** 2))])
    U[0,0] /= H[0]
    U[1,0] /= H[0]
    U[0,1] /= H[1]
    U[1,1] /= H[1]
    #make them unit vectors 
    #u1^2 + u2^2 = h^2
    #u1/h and u2/h

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

#if a is symetric...
if A[0,1] == A[1,0]:
    print('This matrix is symmetric')
    if calcVDVT(U, lamda, A):
        exit()
    elif calcVDVT(np.array([[U[0,1],U[0,0]],[U[1,1],U[1,0]]]), lamda, A):
        exit()
    else:
        print("Somthing went wrong computing VDVt")