# Basic-neural-networks
Training asingle layer neural network 3. Create your first ML model Consider the following sets of numbers. Can you see the relationship between them?

X : -1 0 1 2 3 4 Y : -2 1 4 7 10 13

As you look at them, you might notice that the value of X is increasing by 1 as you read left to right and the corresponding value of Y is increasing by 3. You probably think that Y equals 3X plus or minus something. Then, you'd probably look at the 0 on X and see that Y is 1, and you'd come up with the relationship Y=3X+1.

That's almost exactly how you would use code to train a model to spot the patterns in the data!

Now, look at the code to do it.

How would you train a neural network to do the equivalent task? Using data! By feeding it with a set of X's and a set of Y's, it should be able to figure out the relationship between them.

Start with your imports. Here, you're importing TensorFlow and calling it tf for ease of use.

Next, import a library called numpy, which represents your data as lists easily and quickly.

The framework for defining a neural network as a set of sequential layers is called keras, so import that, too.

Define and compile the neural network Next, create the simplest possible neural network. It has one layer, that layer has one neuron, and the input shape to it is only one value. model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])]) Next, write the code to compile your neural network. When you do so, you need to specify two functions—a loss and an optimizer.

In this example, you know that the relationship between the numbers is Y=3X+1.

When the computer is trying to learn that, it makes a guess, maybe Y=10X+10. The loss function measures the guessed answers against the known correct answers and measures how well or badly it did.

Next, the model uses the optimizer function to make another guess. Based on the loss function's result, it tries to minimize the loss. At this point, maybe it will come up with something like Y=5X+5. While that's still pretty bad, it's closer to the correct result (the loss is lower).

The model repeats that for the number of epochs, which you'll see shortly.

Provide the data Next, feed some data. In this case, you take the six X and six Y variables from earlier. You can see that the relationship between those is that Y=3X+1, so where X is -1, Y is -2.

A python library called NumPy provides lots of array type data structures to do this. Specify the values as an array in NumPy with np.array[].

xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float) ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float) Now you have all the code you need to define the neural network. The next step is to train it to see if it can infer the patterns between those numbers and use them to create a model.

Train the neural network The process of training the neural network, where it learns the relationship between the X's and Y's, is in the model.fit call. That's where it will go through the loop before making a guess, measuring how good or bad it is (the loss), or using the optimizer to make another guess. It will do that for the number of epochs that you specify. When you run that code, you'll see the loss will be printed out for each epoch.
model.fit(xs, ys, epochs=500)

You have a model that has been trained to learn the relationship between X and Y. You can use the model.predict method to have it figure out the Y for a previously unknown X. For example, if X is 10, what do you think Y will be? Take a guess before you run the following code:

print(model.predict([10.0])) You might have thought 31, but it ended up being a little over. Why do you think that is?

Neural networks deal with probabilities, so it calculated that there is a very high probability that the relationship between X and Y is Y=3X+1, but it can't know for sure with only six data points. The result is very close to 31, but not necessarily 31.

As you work with neural networks, you'll see that pattern recurring. You will almost always deal with probabilities, not certainties, and will do a little bit of coding to figure out what the result is based on the probabilities, particularly when it comes to classification.
