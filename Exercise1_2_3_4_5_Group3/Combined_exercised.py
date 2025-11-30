
#  necessary imports for the operations to carry out

import numpy as np


import numpy as np
import neurolab as nl



""" Exercise : 1 Generating the training and testing data for the model"""

np.random.seed(1)

nd_Group3 = np.random.uniform(low=-0.6, high=0.6, size=(10, 2))
output_Group3 = np.sum(nd_Group3, axis=1).reshape(10, 1)
print("Training Input Shape:", nd_Group3.shape)
print("Training Output Shape:", output_Group3.shape)
print("\nSample Input:\n", nd_Group3)
print("\nSample Output (x1 + x2):\n", output_Group3)
print("\n---- The training set (input) ----")
print(nd_Group3)

# Single layer: 2 inputs, 6 neurons, 1 output
net1 = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6]], [6, 1])


print(" single-layer Training")
 
error1 = net1.train(
    nd_Group3, 
    output_Group3,
    show=15,
    goal=0.00001
)
test_value = [[0.1, 0.2]]
result_1 = net1.sim(test_value)
print("\nResult #1 Single Layer Output for 0.1 + 0.2:")
print(result_1)


#exercise 2
# Two hidden layers
net2 = nl.net.newff(
    [[-0.6, 0.6], [-0.6, 0.6]],
    [5, 3, 1]
)
net2.trainf = nl.train.train_gd
print(" multi-layer Training")
error2 = net2.train(
    nd_Group3,
    output_Group3,
    epochs=1000,
    show=100,
    goal=0.00001
)

# Testing
result_2 = net2.sim(test_value)
print("\nResult #2 (Multi-Layer Output for 0.1 + 0.2):")
print(result_2)














""" Exercise : #3 Generating the training and testing data for the model"""

"""Generating the training data for the neural network model.

Generating the training random data using the the uniform method

Setting the size as 100x2 as per  the requirement   (Generating 100 random instances in 2 sets )"""

input_Group_Ex3  = np.random.uniform(low=-0.6 , size=(100,2), high=0.6);


"""Creating the testing (output) data set for from the training data set"""
output_Group3_Ex3 = np.sum(input_Group_Ex3 , axis=1).reshape(100,1)


print(output_Group3_Ex3)

