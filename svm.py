import numpy as np
import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import csv
import time


X = []
Y = []
Xq1 = []
Yq1 = []

start1 = time.time()
with open("mnist/train.csv") as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		Y.append(float(row[784]))
		X.append(temp)
		if(float(row[784])==5):
			Yq1.append(1)
			Xq1.append(temp)
		if(float(row[784])==6):
			Yq1.append(-1)
			Xq1.append(temp)

end1 = time.time()
print("Input done, Time taken", end1-start1)
start2 = time.time()
alpha_count = len(Xq1)
# alpha_count = 10
#######################################################
q1 = np.ones((alpha_count, 1))*-1
#######################################################
# P1 = np.zeros((alpha_count, alpha_count))
# for i in range(alpha_count):
# 	for j in range(i, alpha_count):
# 		P1[i][j]=Yq1[i]*Yq1[j]*np.inner(Xq1[i],Xq1[j])
		# P1[j][i] = P1[i][j]
matX = np.array(Xq1)
matY = np.array([Yq1])
P1 = np.inner(matX, matX)
P1 = P1*((matY.transpose()).dot(matY))
# print(matY)

# print("inner product done")
#######################################################
A1 = np.zeros((1,alpha_count))
for i in range(alpha_count):
	A1[0][i] = Yq1[i] 

#######################################################
temp1 = np.identity(alpha_count)*(-1)
temp2 = np.identity(alpha_count)
G1 = np.concatenate((temp1, temp2), axis=0)

#######################################################
C = 1.0
temp1 = np.zeros((alpha_count,1))
temp2 = np.ones((alpha_count,1))*C
h1 = np.concatenate((temp1, temp2), axis=0)
#######################################################
# b1 = np.array([[0]])
b1 = np.zeros((1,1))
# print(b1.shape)

P = matrix(P1)
q = matrix(q1)
G = matrix(G1)
h = matrix(h1)
A = matrix(A1)
b = matrix(b1)


end2 = time.time()
print("Matrices Made, Time taken", end2-start2)

solvers.options["show_progress"] = False

start3 = time.time()
solution = solvers.qp(P,q,G,h,A,b)
print(solution['status'])
print(solution['x'])
print(solution['primal objective'])

end3 = time.time()
print("Solved, Time taken", end3-start3)
