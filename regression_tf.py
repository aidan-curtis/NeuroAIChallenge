import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
rng = numpy.random
import numpy as np
import math



costs = []
# for tt in range(1, 100, 5):
    # Parameters
learning_rate = 0.001
training_epochs = 20000
display_step = 50

N = 20
I = 500
O = 8



# Training Data
train_X = np.transpose(np.tile(np.arange(0, 10, .1), (N, 1)))

print(train_X)
train_Y = np.transpose(np.array([
    np.array([p**2 for p in np.arange(0, 10, .1)]),
    np.array([p**3-10*p**2+p-1 for p in np.arange(0, 10, .1)]),
    np.array([p**(3/2)-20*p**(0.5)+2*p+2 for p in np.arange(0, 10, .1)]),
    np.array([3*p**(5/2)-20*p**(0.3)-10*p+5 for p in np.arange(0, 10, .1)]),
    [math.sin(math.pi*p) for p in np.arange(0, 10, .1)],
    [math.cos(math.pi*p) for p in np.arange(0, 10, .1)],
    [math.sin(2*math.pi*p) for p in np.arange(0, 10, .1)],
    np.array([math.tan(math.pi * (p+0.501)) for p in np.arange(0, 10, .1)])/1000.0]))



#Normalize output



train_Y = np.clip(train_Y, -1000, 1000)


n_samples = train_X.shape[0]

# tf Graph Input
X = tf.placeholder(tf.float32, shape = (None, N))
Y = tf.placeholder(tf.float32, shape = (None, O))

# Set model weights




w1 = tf.Variable(np.random.uniform(-1,1, [I,N]).astype(np.float32), name="weight1")
b0 = tf.Variable(np.random.uniform(-1,1, [1,N]).astype(np.float32), name="bias0")
b1 = tf.Variable(np.random.uniform(-1,1, [1,I]).astype(np.float32), name="bias1")
w2 = tf.Variable(np.random.uniform(-1,1, [O,I]).astype(np.float32), name="weight2")
b2 = tf.Variable(np.random.uniform(-1,1, [1,O]).astype(np.float32), name="bias2")


print(b0)
# Construct a linear model
pred = tf.add(tf.tanh(tf.transpose(tf.matmul(w1, tf.transpose(tf.add(X, b0))))), b1)


layer2 = tf.add(tf.transpose(tf.matmul(w2, tf.transpose(pred))), b2)
test = layer2

# Mean squared error
C = 0.01
cost = tf.reduce_mean(tf.square(layer2-Y))+C*tf.reduce_sum(tf.square(w2))

# Gradient descent

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()




def plot_model_output(model_output):

    f, axarr = plt.subplots(2, 4)

    actual_output = np.array([p**2 for p in np.arange(0, 10, .01)])
    axarr[0,0].plot(model_output[:,0], label='Model Output')
    axarr[0,0].plot(actual_output, label= "Actual Output")

    actual_output = np.array([p**3-10*p**2+p-1 for p in np.arange(0, 10, .01)])
    axarr[0,1].plot(model_output[:,1], label='Model Output')
    axarr[0,1].plot(actual_output, label= "Actual Output")

    actual_output = np.array([p**(3/2)-20*p**(0.5)+2*p+2 for p in np.arange(0, 10, .01)])
    axarr[0,2].plot(model_output[:,2], label='Model Output')
    axarr[0,2].plot(actual_output, label= "Actual Output")

    actual_output = np.array([3*p**(5/2)-20*p**(0.3)-10*p+5 for p in np.arange(0, 10, .01)])
    axarr[0,3].plot(model_output[:,3], label='Model Output')
    axarr[0,3].plot(actual_output, label= "Actual Output")

    actual_output = np.array([math.sin(math.pi*p) for p in np.arange(0, 10, .01)])
    axarr[1,0].plot(model_output[:,4], label='Model Output')
    axarr[1,0].plot(actual_output, label= "Actual Output")

    actual_output = np.array([math.cos(math.pi*p) for p in np.arange(0, 10, .01)])
    axarr[1,1].plot(model_output[:,5], label='Model Output')
    axarr[1,1].plot(actual_output, label= "Actual Output")

    actual_output = np.array([math.sin(2*math.pi*p) for p in np.arange(0, 10, .01)])
    axarr[1,2].plot(model_output[:,6], label='Model Output')
    axarr[1,2].plot(actual_output, label= "Actual Output")

    actual_output = np.array([math.tan(math.pi * (p+0.5)) for p in np.arange(0, 10, .01)])
    axarr[1,3].plot(model_output[:,7], label='Model Output')
    axarr[1,3].plot(actual_output, label= "Actual Output")

    plt.show()


# Start training
with tf.Session() as sess:
    sess.run(init)
    c = 0
    # Fit all training data
    for epoch in range(training_epochs):
        # for (x, y) in zip(train_X, train_Y):
        sess.run(optimizer, feed_dict={X: train_X, Y: train_Y})



        #Display logs per epoch step
        if (epoch+1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))


    costs.append(c)
    print(c)
    test_X = np.transpose(np.tile(np.arange(0, 10, .01), (N, 1)))
    out = sess.run(layer2, feed_dict={X: test_X})
    plot_model_output(out)
    print("Optimization Finished!")


# print(costs)
# plt.plot(costs)
# plt.ylabel('cost')
# plt.show()





