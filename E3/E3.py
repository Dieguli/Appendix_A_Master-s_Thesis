import numpy as np
from scipy import stats
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
from tensorflow.python.tools import freeze_graph


# PREPARING DATA AND PREPROCESSING
columns = ['channel_1','channel_2','channel_3', 'channel_4']

class1_data = pd.read_csv('class1_200_notch.txt',sep = " ", header = None, names = columns)
class2_data = pd.read_csv('class3_200_notch.txt',sep = " ", header = None, names = columns)
class3_data = pd.read_csv('class4_200_notch.txt',sep = " ", header = None, names = columns)


N_TIME_STEPS = 200
N_FEATURES = 4 
step = 200
segments = []
labels = []
print(len(class1_data)) 

for i in range(0, len(class1_data) - 199, step):
    ch1 = class1_data['channel_1'].values[i: i + N_TIME_STEPS]
    ch2 = class1_data['channel_2'].values[i: i + N_TIME_STEPS]
    ch3 = class1_data['channel_3'].values[i: i + N_TIME_STEPS]
    ch4 = class1_data['channel_4'].values[i: i + N_TIME_STEPS]
    segments.append([ch1, ch2, ch3, ch4])
    y= [1.0, 0.0, 0.0]
    labels.append(y)

for i in range(0, len(class1_data) - 199, step):
    ch1 = class2_data['channel_1'].values[i: i + N_TIME_STEPS]
    ch2 = class2_data['channel_2'].values[i: i + N_TIME_STEPS]
    ch3 = class2_data['channel_3'].values[i: i + N_TIME_STEPS]
    ch4 = class2_data['channel_4'].values[i: i + N_TIME_STEPS]
    segments.append([ch1, ch2, ch3, ch4])
    y= [0.0, 1.0, 0.0]
    labels.append(y)

for i in range(0, len(class1_data) - 199, step):
    ch1 = class3_data['channel_1'].values[i: i + N_TIME_STEPS]
    ch2 = class3_data['channel_2'].values[i: i + N_TIME_STEPS]
    ch3 = class3_data['channel_3'].values[i: i + N_TIME_STEPS]
    ch4 = class3_data['channel_4'].values[i: i + N_TIME_STEPS]
    segments.append([ch1, ch2, ch3, ch4])
    y= [0.0, 0.0, 1.0]
    labels.append(y)

print(np.shape(segments))
data = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
print(data.shape)

data = np.asarray(segments, dtype= np.float32).reshape(-1, N_TIME_STEPS, N_FEATURES)
print(data.shape)

RANDOM_SEED = 42
labels = np.array(labels)
print(labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=RANDOM_SEED)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

#------------------------------------------
N_CLASSES = 3 
N_HIDDEN_UNITS = 64 

#SAVING AND PREPARING MODEL FOR ANDROID

save_path = "./RNN_freeze/"

saver=tf.train.Saver()
model_save = save_path + "RNN.ckpt"

# Freeze the graph
MODEL_NAME = 'RNN' #name of the model optional
input_graph_path = save_path+'savegraph_RNN.pbtxt'#complete path to the input graph
checkpoint_path = save_path+'RNN.ckpt' #complete path to the model's checkpoint file
input_saver_def_path = ""
input_binary = False
#output_node_names = "output" #output node's name. Should match to that mentioned in your code
output_node_names = "y_"
restore_op_name = "save/restore_all"
filename_tensor_name = "save/Const:0"
output_frozen_graph_name = save_path+'frozen_model_'+MODEL_NAME+'.pb' # the name of .pb file you would like to give
clear_devices = True


#MODEL 2
def LSTM_Network(input):

#Dense layer
    W = {
        'output': tf.Variable(tf.random_normal([N_HIDDEN_UNITS, N_CLASSES]))
    }
    biases = {
        
        'output': tf.Variable(tf.random_normal([N_CLASSES]))
    }

    x = tf.unstack(input, N_TIME_STEPS, 1)

    #Stack 2 LSTM layers

    lstm_layers = [tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0) for _ in range(2)] #range(2)
    lstm_layers = tf.contrib.rnn.MultiRNNCell(lstm_layers)

    """
    layer_1 = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0)
    layer_2 = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS, forget_bias=1.0)

    outputs_1, _ = tf.contrib.rnn.static_rnn(layer_1, x, dtype=tf.float32)
    dropout_1 = tf.layers.dropout(outputs_1, rate = 0.2)
    outputs_2, _ = tf.contrib.rnn.static_rnn(layer_1, dropout_1, dtype=tf.float32)
    dropout_2 = tf.layers.dropout(outputs_2, rate = 0.2)
    """

    outputs, _ = tf.contrib.rnn.static_rnn(lstm_layers, x, dtype=tf.float32)
    dropout_1 = tf.layers.dropout(outputs, rate = 0.2)
    #lstm_last_output = outputs[-1]
    #lstm_last_output = dropout_2[-1]
    lstm_last_output = dropout_1[-1]

    return tf.matmul(lstm_last_output, W['output']) + biases['output']

#-------------------------------------------------------------------------------------------

tf.reset_default_graph()

X = tf.placeholder(tf.float32, [None, N_TIME_STEPS, N_FEATURES], name="input")
# [time_steps, batch_size, num_features]
Y = tf.placeholder(tf.float32, [None, N_CLASSES])

pred_Y = LSTM_Network(X)
pred_softmax = tf.nn.softmax(pred_Y, name="y_")

#using L2 regularization for minimizing the loss

L2_LOSS = 0.0015
#L2_LOSS = 0.000015
L2 = L2_LOSS * \
    sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred_Y, labels= Y)) + L2

#Defining the optimizer for the model

#LEARNING_RATE = 0.0025
LEARNING_RATE = 0.0025

optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

correct_pred = tf.equal(tf.argmax(pred_softmax, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, dtype=tf.float32))

#Training the model



#saver = tf.train.Saver()

history = dict(train_loss = [], train_acc = [], test_loss = [], test_acc = [])

N_EPOCHS = 100
BATCH_SIZE = 20
#train_count =100
train_count =len(x_train)
best_acc = 0

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    train_count = len(x_train) #number of rows
    for i in range(1, N_EPOCHS + 1):
        for start, end in zip(range(0, train_count, BATCH_SIZE), range(BATCH_SIZE, train_count + 1, BATCH_SIZE)):
            sess.run(optimizer, feed_dict={X:x_train[start:end],
                                       Y:y_train[start:end]})
        _, acc_train, loss_train = sess.run([pred_softmax, accuracy, loss], feed_dict={X: x_train, Y:y_train})
        _, acc_test, loss_test = sess.run([pred_softmax, accuracy, loss], feed_dict={X: x_test, Y:y_test})
        history['train_loss'].append(loss_train)
        history['train_acc'].append(acc_train)
        history['test_loss'].append(loss_test)
        history['test_acc'].append(acc_test)
        if(acc_test>best_acc):
            best_acc = acc_test

        print("test accuracy in history {0:f}".format(acc_test))
        print("test loss in history {0:f}".format(loss_test))
        print("epoch = " + str(i))
    predictions, acc_final, loss_final = sess.run([pred_softmax, accuracy, loss], feed_dict={X: x_test, Y:y_test})
    print()
    print("Final Results: Accuracy: {0:.2f}, Loss: {1:.2f}".format(acc_final,loss_final))
    print("Parameters = N_HIDDEN_UNITS : " + str(N_HIDDEN_UNITS) + " N_EPOCHS : " + str(N_EPOCHS))
    print("Best accuracy = " + str(best_acc) )

    print("training finished ") 
    
    #1)
	saver.save(sess,model_save) #checkpoint_path built
	tf.train.write_graph(sess.graph_def, save_path, 'savegraph_RNN.pbtxt') # input_graph_path built

	#2)
	freeze_graph.freeze_graph(input_graph_path, input_saver_def_path,input_binary, checkpoint_path, output_node_names,restore_op_name, filename_tensor_name,output_frozen_graph_name, clear_devices, "")
    
    plt.plot(history['train_loss'], 'k-', label='Train Loss')
    plt.plot(history['test_loss'], 'r--', label='Test Loss')
    plt.title('Loss (MSE) per Epoch')
    plt.legend(loc='upper right')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
