
#  necessary imports for the operations to carry out

import numpy as np



""" Exercise : 1 Generating the training and testing data for the model"""

"""Generating thee training data for the neural network model.

Generating the training random data using the the uniform method

Setting the size as 10x2 as per  the requirement"""

nd_Group3  = np.random.uniform(low=-0.6 , size=(10,2), high=0.6);


"""Printing the values of the array for the general insight """

print("\n----The training set (input)---\n",nd_Group3.shape);
print(nd_Group3)


"""Creating the testing (output) data set for from the training data set"""







