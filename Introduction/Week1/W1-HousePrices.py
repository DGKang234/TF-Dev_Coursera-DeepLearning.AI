import numpy as np
from tensorflow import keras

def house_model(y_new):
    """
    :param y_new: the data that we want to predict
    xs: 1D array which has 1 ~ 10 digits
    ys: 1D array which has (xs+1)*0.5 digits

    Training:
    A Sequential model is appropriate for a plain stack of layers where each
    layer has exactly one input tensor and one output tensor.

    optimizer = stochastic gradient descent
    loss =  mean squared error

    fitting xs to ys with 100 iterations

    :return: predict the value for the y_new[0]
    """

    xs = np.array([x for x in range(10)])
    ys = np.array([(x+1.0) * 0.5 for x in range(10)])

    model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer = 'sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs = 100)

    return model.predict(y_new[0])


 if __name__ == "__main__":
     prediction = house_moel([7.0])
     print(prediction)

# expected output ~ 19
# e.g. [[18.984243]]