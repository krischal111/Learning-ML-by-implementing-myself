# I am implementing a simple regression model:
# This is univariate linear regression model (univariate LRM)
# single output single input: y = a + bx

from random import random 
from copy import copy

def usemodel( parameters , x ):
    a = parameters[0]
    b = parameters[1]
    y = a + b * x
    return y

# I don't think we need this, do we?
def loss_single_output(expected_output, our_output):
    loss = (expected_output - our_output)**2
    return loss

# Just getting an idea. We actually need the gradient function.
def loss_function(parameters, list_of_expected_output, list_of_our_output):
    m = len(list_of_expected_output)

    total_loss = 0
    for (expected_output, our_output) in zip(list_of_expected_output, list_of_our_output):
        total_loss += loss_single_output(expected_output, our_output)
    
    total_loss = total_loss / (2*m)

# Gradient function:
def loss_gradient_single_output(parameters, single_input, single_output):
    # gradient of loss function with respect to parameters:
    # for a: 
    x = single_input
    y = single_output
    a = parameters[0]
    b = parameters[1]
    a_grad = (a + x * b - y)
    b_grad = (a + x * b - y) * x

    return [a_grad, b_grad]

# Gradient function:
def loss_gradient(parameters, list_of_inputs, list_of_outputs):
    m = len(list_of_inputs)

    total_a_grad = 0
    total_b_grad = 0

    for (input, output) in zip(list_of_inputs, list_of_outputs):
        single_loss = loss_gradient_single_output(parameters, input, output)
        total_a_grad += single_loss[0]
        total_b_grad += single_loss[1]
    
    total_a_grad = total_a_grad / (2*m)
    total_b_grad = total_b_grad / (2*m)

    return [total_a_grad, total_b_grad]

# machine learning
def learn_a_little_bit(parameter, list_of_inputs, list_of_outputs, learning_rate):
    loss_grad = loss_gradient(parameter, list_of_inputs, list_of_outputs)
    print([i*learning_rate for i in loss_grad])
    new_parameter = copy(parameter)

    for i in range(0, len(parameter)):
        new_parameter[i] = parameter[i] - learning_rate * loss_grad[i]
    
    return new_parameter

# input database and learn it

# count = int(input("Please input how many datasets are there = "))

inputs = [1, 2, 3, 4, 5]
outputs = [3, 5, 7, 9, 11]

# for i in range(count):
    # inputs.append(float(input(f"Input the {i+1}th input data = ")))
    # outputs.append(float(input(f"Input the {i+1}th output data = ")))

print(f"Input datas = {inputs}")
print(f"Output datas = {outputs}")


# Now I am applying a gradient descent.

# Start from random data
# Apply grad descent from sequence of inputs vs outputs
# Apply correction
parameter = [random(), random()]
# parameter = [1.1, 2.1]
print(f"Initial parameters:")
print(f" y = {parameter[0]} + {parameter[1]} x")

# set the learning rate
learning_rate = .04

# learning for a single cycle
print(f"After learning a little bit:")
parameter = learn_a_little_bit(parameter, inputs, outputs, learning_rate)
print(f" y = {parameter[0]} + {parameter[1]} x")

# learning for 10 cycles
print(f"After learning for 10 cycles : ")
for _ in range(10):
    parameter = learn_a_little_bit(parameter, inputs, outputs, learning_rate)
print(f" y = {parameter[0]} + {parameter[1]} x")


# learning for 100 cycles
# print(f"After learning for 100 cycles : ")
# for _ in range(100):
    # parameter = learn_a_little_bit(parameter, inputs, outputs, learning_rate)
# print(f" y = {parameter[0]} + {parameter[0]} x")
