import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import csv
import time
from numpy import linalg as la
import math
from svmutil import *
import sys
import random
# X = []
# Y = []
# X_test = []
# Y_test = []

# train = sys.argv[1]
# test = sys.argv[2]
# part = sys.argv[3]

# time_gap = float(sys.argv[4])



X = []
Y = []
X_test = []
Y_test = []

# num = 5

start1 = time.time()

train = "mnist/train.csv"
test = "mnist/test.csv"

with open(train) as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		Y.append(float(row[784]))
		X.append(temp)


with open(test) as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		Y_test.append(float(row[784]))
		X_test.append(temp)

end1 = time.time()
print("Input done, Time taken(sec)", int(end1-start1))
start2 = time.time()
# alpha_count = len(Xq1)

part = 'c'

C = ['-t 2 -c 1e-5 -b 0 -g 0.05 -q',
	 '-t 2 -c 1e-3 -b 0 -g 0.05 -q', 
	 '-t 2 -c 1    -b 0 -g 0.05 -q',
	 '-t 2 -c 5    -b 0 -g 0.05 -q',
	 '-t 2 -c 10   -b 0 -g 0.05 -q' ]


for i in range(5):
	prob  = svm_problem(Y, X)
	param = svm_parameter('-t 2 -c 1 -b 0 -g 0.05 -q')
	m = svm_train(prob, param, '-q')
	p_label, p_acc, p_val = svm_predict(Y_test, X_test, m, '-b 0 -q')
	# print("Accuracy using LIBSVM: ", p_acc)
	ACC, MSE, SCC = evaluations(Y_test, p_label)
	print("Accuracy using LIBSVM: ", ACC)
	# alpha_libsvm = m.get_sv_coef()
	# SV_indices = m.get_sv_indices()
	# alpha_svm = []
	# j=0
	# for i in range(len(Yq1_test)):
	# 	if(i==SV_indices[j]):
	# 		alpha_svm.append(alpha_libsvm[j][0])
	# 		j+=1
	# 	else:
	# 		alpha_svm.append(0)
