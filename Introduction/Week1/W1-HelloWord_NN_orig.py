import numpy as np
import keras

# dense: define the layer of connected neurons 1 "Dense" so only one layer here
# one unit in it so single neuron, input shape -> one value
# successive layer is defined in sequence
model = keras.Sequential([
                        keras.layers.Dense(units=1, input_shape=[1])
])

# sgd = stochastic gradient descent
model.compile(optimizer='sgd', loss='mean_squared_error')

# represent the known data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

##RQ
#xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
#ys = np.array([1, 1, 6, 38, 233, 21910], dtype=float)

##GA
#xs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
#ys = np.array([0, 0, 5, 39, 239, 1718], dtype=float)

#PBEsol
#xs = np.array([1, 2, 3, 4, 5, 6], dtype=int)
#ys = np.array([1, 1, 5, 38, 233, 2316], dtype=int)

# fit the x values to the y values, training loop 500 times
model.fit(xs, ys, epochs=1000)

# expected results is ~ 19
print(model.predict([7]))
print(model.predict([8]))
print(model.predict([9]))
print(model.predict([10]))



