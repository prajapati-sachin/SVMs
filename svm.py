import numpy as np
import numpy as np
import cvxopt
import csv


X = []
Y = []

with open("mnist/train.csv") as fileX:
	x_reader = csv.reader(fileX)
	for row in x_reader:
		temp = []
		for i in range(784):
			temp.append(float(row[i])/255)
		Y.append(float(row[784]))
		X.append(temp)



