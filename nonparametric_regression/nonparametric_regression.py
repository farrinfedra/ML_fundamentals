import math
import matplotlib.pyplot as plt
import numpy as np

data = np.genfromtxt('hw04_data_set.csv', delimiter = ",", skip_header = 1)

#print(data)

#first 150 train and rest is test set x
x_train = data[0:150, 0]
x_test = data[150:, 0]

#first 150 train and rest is test set x
y_train = data[0:150, 1].astype(int)
y_test = data[150:, 1].astype(int)

minimum_value = 1.5
maximum_value = 5.2

data_interval = np.round(np.linspace(minimum_value, maximum_value, 3701), decimals = 3)

bin_width = 0.37
left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)

g_hat = np.asarray([np.mean( y_train[(left_borders[b] < x_train) & (right_borders[b] >= x_train)]) for b in range(len(left_borders))])

plt.figure(figsize = (12, 6))
plt.plot(x_train, y_train, "bo", label = "training")
plt.plot(x_test, y_test, "ro", label = "test")

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [g_hat[b], g_hat[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [g_hat[b], g_hat[b + 1]], "k-")

plt.title(f"h = {bin_width}")
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time to next eruption (min)')
plt.legend()
plt.show()

#Calculate RMSE of regressogram for test data points
RMSE = np.sqrt(np.mean([(y_test[i] - g_hat[(left_borders < x_test[i]) & (right_borders >= x_test[i])])**2 for i in range(len(y_test))]))

print(f"Regressogram => RMSE is {RMSE} when h is {bin_width}")

#Learn a running mean smoother
bin_width = 0.37

g_hat = np.asarray([np.mean(y_train[((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))]) for x in data_interval])

#Draw training data points, test data points, and running mean smoother in the same figure
plt.figure(figsize = (12, 6))
plt.plot(x_train, y_train, "bo", label = "training")
plt.plot(x_test, y_test, "ro", label = "test")
plt.plot(data_interval, g_hat, "k-")

plt.title(f"h = {bin_width}")
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time to next eruption (min)')
plt.legend()
plt.show()


#Calculate the RMSE of running mean smoother for test data points
RMSE = np.sqrt(np.mean((y_test - np.concatenate([g_hat[np.where(data_interval == x)] for x in x_test])) ** 2))
print(f"Running Mean Smoother => RMSE is {RMSE} when h is {bin_width}")


#Learn a kernel smoother by setting the bin width parameter to 0.37
bin_width = 0.37
g_hat = np.asarray([np.sum(np.exp(-0.5 * (x - x_train)**2 / bin_width**2)*y_train) / np.sum(np.exp(-0.5 * (x - x_train)**2 / bin_width**2))
                    for x in data_interval])

#Draw training data points, test data points, and kernel smoother

plt.figure(figsize = (12, 6))
plt.plot(x_train, y_train, "bo", label = "training")
plt.plot(x_test, y_test, "ro", label = "test")

plt.plot(data_interval, g_hat, "k-")

plt.title(f"h = {bin_width}")
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting time to next eruption (min)')
plt.legend()
plt.show()

#Calculate the RMSE of kernel smoother for test data points

RMSE = np.sqrt(np.mean((y_test - np.concatenate([g_hat[np.where(data_interval == x)] for x in x_test])) ** 2))
print(f"Kernel Smoother => RMSE is {RMSE} when h is {bin_width}")
