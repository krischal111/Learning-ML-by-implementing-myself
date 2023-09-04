# LRM = linear regression model
# the main equation is y = a + bx, where x is the input and y is the output
# We plug in x to get y, is all the model does.

from random import random
# from pyprobs import

# Now to check the model's performance, we create various functions

# First of all, we need to use the model: this returns y = a + b x
def model_use(model, x):
    a, b = model
    return a + b * x

# then we need to check the derivative of the loss, but we apply back propragation and separate that to parts
def part_diff_first(model, x, y):
    residue = model_use(model, x) - y
    return residue

def part_diff_param_a(model, x, y):
    return 1

def part_diff_param_b(model, x, y):
    return x

def loss_gradient_single_example(model, x, y):
    first = part_diff_first(model, x, y)
    second_a = part_diff_param_a(model, x, y)
    second_b = part_diff_param_b(model, x, y)
    return [first * second_a, first * second_b]

def loss_gradient_all_example(model, inputs, outputs):
    m = len(inputs)
    assert(len(inputs) == len(outputs))
    loss_gradients = [0,0]
    for (x, y) in zip(inputs, outputs):
        gradients = loss_gradient_single_example(model, x, y)
        loss_gradients = [prev_grad_sum + new_grad for (prev_grad_sum, new_grad) in zip(loss_gradients, gradients)]
    loss_gradients = [lg / m for lg in loss_gradients]
    return loss_gradients

def learn_a_little_bit(model, inputs, outputs, learning_rate):
    loss_gradients = loss_gradient_all_example(model, inputs, outputs)
    return [model_parameter - learning_rate * lg for (model_parameter, lg) in zip(model, loss_gradients)]

def print_model(model):
    print(f"Model is : y = {model[0]} + {model[1]} x")

inputs = [1, 2, 3, 4, 5]
outputs = [3, 5, 7, 9, 11]

model = [random(), random()]
learning_rate = 0.1

print("Before learning, the model is ")
print_model(model)

iter_count = 1000
print(f"After {iter_count} iterations ")
for _ in range(iter_count):
    model = learn_a_little_bit(model, inputs, outputs, learning_rate)
    print_model(model)
