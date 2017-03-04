#initialize the weights of the neural network
import numpy as np
import matplotlib.pyplot as plt

X_file = np.genfromtxt('mpg.csv', delimiter=',', skip_header=1)
# N equals a one (0) dimensional array based on the size of X_file column 0
N = np.shape(X_file)[0]
# X = stacked array of ones and business in the weights, columns(N), rows(1s, weight)
X = np.hstack((np.ones(N).reshape(N, 1), X_file[:,4].reshape(N, 1)))
print(X)
#print(X_file[:,4])
# Y = X_file[:, 0]
#
# #Standardize inputs, subtrating mean div by std deviation
# X[:,1]= (X[:,1]-np.mean(X[:, 1]))/np.std(X[:,1])
#
# w = np.array([0, 0])
# max_iter = 100
# eta = 1E-3
# for t in range(0, max_iter):
#     grad_t = np.array([0.,0.])
#     for i in range(0, N):
#         x_i = X[i, :]
#         y_i = Y[i]
#         h = np.dot(w, x_i)-y_i
#         grad_t += 2*x_i*h
#
#     w = w - eta*grad_t
# print ("Weights found:",w)
#
# # Plot the data and best fit line
# tt = np.linspace(np.min(X[:, 1]), np.max(X[:, 1]), 10)
# bf_line = w[0]+w[1]*tt
#
# plt.plot(X[:, 1], Y, 'kx', tt, bf_line, 'r-')
# plt.xlabel('Weight (Normalized)')
# plt.ylabel('MPG')
# plt.title('ANN Regression on 1D MPG Data')
#
# plt.savefig('mpg.png')
#
# plt.show()
