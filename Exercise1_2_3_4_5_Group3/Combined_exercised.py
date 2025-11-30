
#  necessary imports for the operations to carry out

import numpy as np









""" Exercise : #3 Generating the training and testing data for the model"""

"""Generating the training data for the neural network model.

Generating the training random data using the the uniform method

Setting the size as 100x2 as per  the requirement   (Generating 100 random instances in 2 sets )"""

input_Group_Ex3  = np.random.uniform(low=-0.6 , size=(100,2), high=0.6);


"""Creating the testing (output) data set for from the training data set"""
output_Group3_Ex3 = np.sum(input_Group_Ex3 , axis=1).reshape(100,1)


print(output_Group3_Ex3)

