from __init__ import *
import numpy as np

# Question 1:
def nevilles_method(x_points, y_points, value):
    # Setting up the matrix
    size = len(x_points)
    matrix = np.zeros((size, size))

    for counter, row in enumerate(matrix):
        row[0] = y_points[counter]
    num_of_points = 2

    for i in range(1, num_of_points):
        for j in range(1, i + 1):
            first_multiplication = (value - x_points[i - j]) * matrix[i][j-1]
            second_multiplication = (value - x_points[i]) * matrix[i-1][j-1]
            denominator = x_points[i] - x_points[i - j]
            # this is the value that we will find in the matrix
            coefficient = (first_multiplication - second_multiplication) / denominator 
            matrix[i][j] = coefficient
    
    print(matrix[num_of_points - 1][num_of_points - 1])
    print()

# Question 2:
def newtons_forward_method(x_val, f_x_val):
    # set up the matrix
    x_val_0 = 7.2
    x_val_1 = 7.4
    x_val_2 = 7.5
    x_val_3 = 7.6
    f_x_val_0 = 23.5492
    f_x_val_1 = 25.3913
    f_x_val_2 = 26.8224
    f_x_val_3 = 27.4589

    # Finding First divided difference
    first_divided_difference = (f_x_val_1 - f_x_val_0) / (x_val_1 - x_val_0)
    first_divided_difference_2 = (f_x_val_2 - f_x_val_1) / (x_val_2 - x_val_1)
    first_divided_difference_3 = (f_x_val_3 - f_x_val_2)/ (x_val_3 - x_val_2)

    # Finding second divided difference
    second_divided_difference = (first_divided_difference_2 - first_divided_difference) / (x_val_2 - x_val_0)
    second_divided_difference_2 = (first_divided_difference_3 - first_divided_difference_2) / (x_val_3 - x_val_1)

    # Finding third divided difference
    third_divided_difference = (second_divided_difference_2 - second_divided_difference) / (x_val_3 - x_val_0)

    # Printing the results
    results = [first_divided_difference, second_divided_difference, third_divided_difference]
    print(results)
    print()

    # Question 3:
    approx_val = 7.3
    # Using results from 3 to approximate f(7.3)
    x = f_x_val_0 + first_divided_difference * (approx_val - x_val_0) + second_divided_difference * (approx_val - x_val_1) * (approx_val - x_val_0)\
        + third_divided_difference * (approx_val - x_val_2) * (approx_val - x_val_1) * (approx_val - x_val_0)
    print(x)
    print()
    
# Question 4:
def divided_difference_method(x, y, dy):
    size = len(x) * 2
    divided_difference_matrix = np.zeros((size, size))

    for i in range(size):
        index = round(i // 2)
        divided_difference_matrix[i][0] = x[index]
        divided_difference_matrix[i][1] = y[index]
        if i + 1 < size:
            divided_difference_matrix[i + 1][2] = dy[index]

    np.set_printoptions(precision=7, suppress=True, linewidth=100)
    print(divided_difference_matrix)
    
# Question 5:
# Fill in data
x_point = [2, 5, 8, 10]
y_point = [3, 5, 7, 9]

# Problem A:
# Set up matrix
n = len(x_point)
matrix_5 = np.zeros((n, n))
matrix_5[0, 0] = 1
matrix_5[n-1, n-1] = 1
for i in range(1, n-1):
    matrix_5[i, i-1] = x_point[i] - x_point[i-1]
    matrix_5[i, i] = 2 * (x_point[i+1] - x_point[i-1])
    matrix_5[i, i+1] = x_point[i+1] - x_point[i]

# Problem B:
vector_b = np.zeros(n)
for i in range(1, n-1):
    vector_b[i] = 3 * (y_point[i+1] - y_point[i]) / (x_point[i+1] - x_point[i]) -\
                3 * (y_point[i] - y_point[i-1]) / (x_point[i] - x_point[i-1])


# Problem C:
vector_x = np.linalg.solve(matrix_5, vector_b)


if __name__ == "__main__":
    # Question 1 setup
    x_points = [3.6, 3.8, 3.9]
    y_points = [1.675, 1.436, 1.318]
    value = 3.7
    nevilles_method(x_points, y_points, value)

    # Question 2 setup
    x_val = [7.2, 7.4, 7.5, 7.6]
    f_x_val = [23.5492, 25.3913, 26.8224, 27.4589]
    newtons_forward_method(x_val, f_x_val)


    # Question 4 output
    x = [3.6, 3.8, 3.9]
    y = [1.675, 1.436, 1.318]
    dy = [-1.195, -1.188, -1.182]
    divided_difference_method(x, y, dy)

# Question 5 outputs   
print(matrix_5)
print()

print(vector_b)
print()

print(vector_x)