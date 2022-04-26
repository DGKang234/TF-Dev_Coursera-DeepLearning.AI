import tensorflow as tf
from tensorflow import keras
print(tf.__version__)
import numpy as np
import matplotlib.pyplot as plt

"""
class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy') >= 0.6): # Experiment with changing this value
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True
"""

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') < 0.4):
            print("\nLoss is low so cancelling training!")
            self.model.stop_Training = True
callbacks = myCallback()

# load data set (fashion_mnist)
# training data set used training the model
# unseen data set, test data set used to evaluate the model
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


index = 0
# set number of characters per row when printing
np.set_printoptions(linewidth=300)
print(f'LABEL: {train_labels[index]}')
print(f'\nIMAEG PIXEL ARRAY:\n {train_images[index]}')
# visualize the image
plt.imshow(train_images[index], cmap='Greys')


# Normalize the pixel values of the train and test images
training_images = train_images / 255.0
test_images = test_images /255.0

# three layer of NN
model = keras.Sequential([
                        keras.layers.Flatten(input_shape=(28, 28)),        # images are 28x28 so this shape is expected to be fed
                        keras.layers.Dense(128, activation=tf.nn.relu),    # hidden layer with 128 neurons, ReLU if x > 0: return x, else: return 0
                        keras.layers.Dense(10, activation=tf.nn.softmax)   # 10 neurons == 10 class of clothing
])                                                                         # https://www.youtube.com/watch?v=LLux1SW--oM&ab_channel=DeepLearningAI
# Increase to 1024 Neurons -- what's the impact?
# Training takes, longer, but is more accurate but that doesn't mean it's always a case of 'more is better'.

# What would happen if you remove the Flatten() layer. Why do you think that's the case?
# it will abort the process with an error. it reinforces the rule of thumb that the first layer in your network should
# be the same shape as your data. Right now our data is 28x28 images, and 28 layers of 28 neurons would be infeasible,
# so it makes more sense to 'flatten' that 28,28 into a 784x1.

# The effect of additional layers in the network won't show significant impact because this is relatively simple data.
# For far more complex data (including color images to be classified as flowers), extra layers are often necessary.

# Try 15 epochs -- you'll probably get a model with a much better loss than the one with 5
# Try 30 epochs -- you might see the loss value stops decreasing, and sometimes increases.
# This is a side effect of something called 'overfitting'

# Declare sample inputs and convert to a tensor
inputs = np.array([[1.0, 3.0, 4.0, 2.0]])
inputs = tf.convert_to_tensor(inputs)
print(f'input to softmax function: {inputs.numpy()}')

# Feed the inputs to a softmax activation function
outputs = tf.keras.activations.softmax(inputs)
print(f'output of softmax function: {outputs.numpy()}')

# Get the sum of all values after the softmax
sum = tf.reduce_sum(outputs)
print(f'sum of outputs: {sum}')

# Get the index with highest value
prediction = np.argmax(outputs)
print(f'class with highest probability: {prediction}')

'''
expected output:
input to softmax function: [[1. 3. 4. 2.]]
output of softmax function: [[0.0320586  0.23688282 0.64391426 0.08714432]]
sum of outputs: 1.0
class with highest probability: 2
'''


model.compile(optimizer = tf.optimizers.Adam(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# execute the training loop
model.fit(training_images, train_labels, epochs=5, callbacks=[callbacks])


# Evaluate the model on unseen data
print(model.evaluate(test_images, test_labels))
# [0.362445205450058, 0.8707000613212585]

#Exercise 1:
classifications = model.predict(test_images)
print(classifications[0])   # It's the probability that this item is each of the 10 class
print(test_labels[0])       # The 10th element on the list is the biggest, and the ankle boot is labelled 9
