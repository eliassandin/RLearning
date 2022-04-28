from numpy import asarray
from numpy import savetxt
import numpy as np
import pandas as pd
from numpy import loadtxt



a = np.array([[1,2,3,4],
	[5,6,7,8]]
	)
b = np.array([[1,2],[3,4]])

#df = pd.DataFrame(a)
#df.to_csv('data.csv', index = False)

file = open('data.csv', 'rb')
data = loadtxt(file,delimiter = ",")
data = data[1:]
all_boards = []
for board in data:
	all_boards.append(board.reshape(( 6, 7)))

file = open('Q_data.csv', 'rb')
Q_data = loadtxt(file,delimiter = ",")
Q_data = Q_data[1:]

print(all_boards[-2])
print(Q_data[-2])
print(all_boards[-1])
print(Q_data[-1])