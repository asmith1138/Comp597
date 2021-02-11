# x = [58, 62, 60, 64, 67, 70] # mother's height
# y = [60, 60, 58, 60, 70, 72] # daughter's height

import numpy as np

class LinearRegression:
    def __init__(self, x_set, y_set):
        self.x_set = x_set
        self.y_set = y_set
        self.alpha = 0.001  # alpha is "learning rate"

    def get_theta(self, theta):
        intercept, slope = theta
        intercept_gradient = 0
        slope_gradient = 0
        m = len(self.y_set)
        for i in range(0, len(self.y_set)):
            x_val = self.x_set[i]
            y_val = self.y_set[i]
            y_predicted = self.get_prediction(slope, intercept, x_val)
            intercept_gradient += (y_predicted - y_val)
            slope_gradient += (y_predicted - y_val) * x_val

        new_intercept = intercept - self.alpha * intercept_gradient
        new_slope = slope - self.alpha * (1/m) * slope_gradient
        return [new_intercept, new_slope]

    def get_prediction(self, slope, intercept, x_val):
        return slope * x_val + intercept

    def calc_cost(self, theta):
        intercept, slope = theta
        sum = 0
        for i in range(0, len(self.y_set)):
            x_val = self.x_set[i]
            y_val = self.y_set[i]
            y_predicted = self.get_prediction(slope, intercept, x_val)
            diff_sq = (y_predicted - y_val) ** 2
            sum += diff_sq

        cost = sum / (2*len(self.y_set))
        return cost

    def iterate(self):
        num_iteration = 0
        current_cost = None
        current_theta = [0, 0]  # initialize to 0

        while num_iteration < 10000:
            if num_iteration % 10 == 0:
                print('current iteration: ', num_iteration)
                print('current cost: ', current_cost)
                print('current theta: ', current_theta)
            new_cost = self.calc_cost(current_theta)
            current_cost = new_cost
            new_theta = self.get_theta(current_theta)
            current_theta = new_theta
            num_iteration += 1

        print(f'After {num_iteration}, total cost is {current_cost}. Theta is {current_theta}')

data = np.genfromtxt("winglength.txt", delimiter=",")
y = data[:,0]
x = data[:,1]
LinearRegression(x,y).iterate()