
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


"""Repeating the steps 4 to 8"""
# Single layer: 2 inputs, 6 neurons, 1 output
net3 = nl.net.newff([[-0.6, 0.6], [-0.6, 0.6]], [6, 1])


print(" single-layer Training")
 
error3 = net3.train(
    input_Group_Ex3, 
    output_Group3_Ex3,
    show=15,
    goal=0.00001
)
test_value = [[0.1, 0.2]]
result_3 = net3.sim(test_value)
print("\nResult #3 Single Layer Output for 0.1 + 0.2:")
print(result_3)








"""Exercise 4"""
""" Repeating the step 1 from exercise 3"""
input_Group3_Ex4  = np.random.uniform(low=-0.6 , size=(100,2), high=0.6);


"""Creating the testing (output) data set for from the training data set"""
output_Group3_Ex4 = np.sum(input_Group_Ex3 , axis=1).reshape(100,1)


"""Two layer neural network with gradinet descent backpropogation network """
import matplotlib.pyplot as plt
net4 = nl.net.newff([[-0.6, 0.6],[-0.6, 0.6]],[5,3,1])
net4.trainf = nl.train.train_gd
error4 = net4.train(input_Group3_Ex4, output_Group3_Ex4, epochs=1000, show=100, goal=0.00001)


"""Plotting the error graph"""
plt.figure()
plt.title("Training Error - Exercise #4")
plt.xlabel("Epochs")
plt.ylabel("Error")
plt.plot(error4)
plt.show()


result4 = net4.sim(test_value);
print("\nResult #4 :")
print(result4)



"""Exercise 5"""
"""Repeating the step of exercise 1 for three inputs"""

# 5A - Single-layer (3 inputs)
input_Group3_Ex5  = np.random.uniform(low=-0.6 , size=(100,3), high=0.6);


"""Creating the testing (output) data set for from the training data set"""
output_Group3_Ex5 = np.sum(input_Group_Ex3 , axis=1).reshape(100,1)


net5 = nl.net.newff([[-0.6, 0.6]]*3,[6,1])
error5 = net5.train(input_Group3_Ex5, output_Group3_Ex5, show=15, goal=0.00001)
result5 = net5.sim([[0.2,0.1,0.2]]);
print("\nResult #5 Single Layer Output for 0.2+0.1 + 0.2 :")
print(result5)


# 5B - Multi-layer (3 inputs)
net6 = nl.net.newff([[-0.6, 0.6]]*3,[5,3,1])
net6.trainf = nl.train.train_gd
error6 = net6.train(input_Group3_Ex5, output_Group3_Ex5, epochs=1000, show=100, goal=0.00001)
result6 = net6.sim([[0.2,0.1,0.2]]);

print("\nResult #6 ")
print(result6)



print("\n--- RESULTS SUMMARY ---")
print("Result #1:", result_1)
print("Result #2:", result_2)
print("Result #3:", result_3)
print("Result #4:", result4)
print("Result #5:", result5)
print("Result #6:", result6)