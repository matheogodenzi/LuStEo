import numpy as np
import matplotlib.pyplot as plt

metrics_evolution = np.loadtxt('deep_network_metrics.txt', delimiter=',')


# Accuracy plot for the training, validation and test datasets
plt.plot(metrics_evolution[:,0])
plt.plot(metrics_evolution[:,2])
plt.plot(metrics_evolution[:,4])
plt.legend(['training', 'validation', 'test'])
plt.xlabel('iterations')
plt.ylabel('accuracy [%]')
plt.title('Accuracy Evolution During Deep Learning Training')
plt.grid()
plt.show()

# f1 score plot for the training, validation and test datasets
plt.plot(metrics_evolution[:,1])
plt.plot(metrics_evolution[:,3])
plt.plot(metrics_evolution[:,5])
plt.legend(['training', 'validation', 'test'])
plt.xlabel('iterations')
plt.ylabel('f1 score')
plt.title('f1 Score Evolution During Deep Learning Training')
plt.grid()
plt.show()


# comparing accuracies for training, validation and test datasets hyperparameters
# self.acc_tr, self.f1_tr, self.acc_val, self.f1_val, self.acc_test, self.f1_test
# neurons(300, 50) lr = 1e-4  iterations = 500 : 91.07763615295481,0.8934765275580588,75.06112469437653,0.6732569470894039,90.1294498381877,0.8845061728464775 almost 300 seconds
# neurons(200, 60) lr = 1e-5  iterations = 500 : 75.55040556199305,0.7050812444687429,71.39364303178485,0.6113329873568146,78.80258899676376,0.7052683376000596 almost 300 seconds
# neurons(200, 60) lr = 1e-4  iterations = 500 : 90.38238702201622,0.8851064828541279,74.81662591687042,0.6684416202324076,88.83495145631068,0.8633273423901765 almost 300 seconds
# neurons(60, 200) lr = 1e-4  iterations = 500 : 90.57551178061027,0.8871263011777724,73.83863080684597,0.6590820725138922,88.67313915857605,0.8632715086737841 182 seconds
# neurons(50, 300) lr = 1e-4  iterations = 500 : 91.8115102356122,0.9042446830355007,75.30562347188264,0.6790791677375211,89.96763754045307,0.88189220411075 178 seconds
# neurons(40, 400) lr = 1e-4  iterations = 500 : 92.19775975280031,0.9069830214474521,74.81662591687042,0.6755072094802385,88.83495145631068,0.8643985532099674 176 seconds
# neurons(60, 400) lr = 1e-4  iterations = 500 : 91.61838547701815,0.9005674017583156,74.81662591687042,0.6719020730961262,89.80582524271844,0.8793455843733604 184 seconds
# neurons(100, 100) lr = 1e-4  iterations = 500 : 90.49826187717265,0.8861586698810462,74.81662591687042,0.670100752786535,87.37864077669903,0.8436771976829622 209 seconds
# neurons(90, 200) lr = 1e-4  iterations = 500 : 91.07763615295481,0.893696614350552,74.81662591687042,0.6723991546634894,88.67313915857605,0.8634697246303025 198 seconds
# neurons(100, 10) lr = 1e-4  iterations = 500 : 90.2665121668598,0.8841427711453314,73.83863080684597,0.6655194974996003,89.15857605177993,0.8743375797834522 208 seconds
