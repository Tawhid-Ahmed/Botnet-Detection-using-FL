# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:06:01 2023

@author: tawhid
"""
import json

import flwr as fl
import pika
import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
import os
import keras
from keras.regularizers import *
from keras.initializers import glorot_uniform

import matplotlib.pyplot as plt
from glob import glob
import seaborn as sns
from PIL import Image
# from imutils import paths
import random
import pickle
import cv2
import datetime
from pprint import pprint
import librosa

import keras.backend as K

K.clear_session()

from keras.models import *
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import *

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc, \
    classification_report
from itertools import cycle
import sys

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8  # Adjust this value as needed
# session = tf.compat.v1.Session(config=config)
#
# tf.profiler.experimental.start('logs/memory_profiler')

current_round = 0

server_url = 'host.docker.internal'

low = int(os.environ.get('low'))
top = int(os.environ.get('top'))
skip = [i for i in range(low, top)]  # args.skip
count = int(os.environ.get('count'))
client_id = int(os.environ.get('client_id'))
dataset_num = int(os.environ.get('dataset_num'))
epochs = int(os.environ.get('epochs'))
batch_size = int(os.environ.get('batch_size'))

print(f'loading {count} rows from dataset number: {dataset_num} with {epochs} epochs...')

fl.common.logger.configure(identifier="myFlowerExperiment", filename=f"log_{client_id}.txt")

benign_df = pd.read_csv('dataset/{num}.benign.csv'.format(num=dataset_num), skiprows=skip, nrows=count)

g_c_df = pd.read_csv('dataset/{num}.gafgyt.combo.csv'.format(num=dataset_num), skiprows=skip, nrows=count)
g_j_df = pd.read_csv('dataset/{num}.gafgyt.junk.csv'.format(num=dataset_num), skiprows=skip, nrows=count)
g_s_df = pd.read_csv('dataset/{num}.gafgyt.scan.csv'.format(num=dataset_num), skiprows=skip, nrows=count)
g_t_df = pd.read_csv('dataset/{num}.gafgyt.tcp.csv'.format(num=dataset_num), skiprows=skip, nrows=count)
g_u_df = pd.read_csv('dataset/{num}.gafgyt.udp.csv'.format(num=dataset_num), skiprows=skip, nrows=count)
m_a_df = pd.read_csv('dataset/{num}.mirai.ack.csv'.format(num=dataset_num), skiprows=skip, nrows=count)
m_sc_df = pd.read_csv('dataset/{num}.mirai.scan.csv'.format(num=dataset_num), skiprows=skip, nrows=count)
m_sy_df = pd.read_csv('dataset/{num}.mirai.syn.csv'.format(num=dataset_num), skiprows=skip, nrows=count)
m_u_df = pd.read_csv('dataset/{num}.mirai.udp.csv'.format(num=dataset_num), skiprows=skip, nrows=count)
m_u_p_df = pd.read_csv('dataset/{num}.mirai.udpplain.csv'.format(num=dataset_num), skiprows=skip, nrows=count)

benign_df['type'] = 'benign'
m_u_df['type'] = 'mirai_udp'
g_c_df['type'] = 'gafgyt_combo'
g_j_df['type'] = 'gafgyt_junk'
g_s_df['type'] = 'gafgyt_scan'
g_t_df['type'] = 'gafgyt_tcp'
g_u_df['type'] = 'gafgyt_udp'
m_a_df['type'] = 'mirai_ack'
m_sc_df['type'] = 'mirai_scan'
m_sy_df['type'] = 'mirai_syn'
m_u_p_df['type'] = 'mirai_udpplain'

data = pd.concat([benign_df, m_u_df, g_c_df,
                  g_j_df, g_s_df, g_t_df,
                  g_u_df, m_a_df, m_sc_df,
                  m_sy_df, m_u_p_df],
                 axis=0, sort=False, ignore_index=True)

# load dataset


# df=pd.read_csv('Combined_file_5.csv')

labels_full = pd.get_dummies(data['type'], prefix='type')
labels_full.head()

data = data.drop(columns='type')
data.head()


def call_mq(ac, ls, c_round, cr):
    connection = pika.BlockingConnection(pika.ConnectionParameters(f'{server_url}'))
    channel = connection.channel()

    # Declare a queue to send the message to
    channel.queue_declare(queue='my_queue')

    # Send a message
    message = {
        'client_id': client_id,
        'accuracy': ac,
        'loss': ls,
        'round': c_round,
        'cr': cr
    }

    json_message = json.dumps(message)

    channel.basic_publish(exchange='', routing_key='my_queue', body=json_message)

    print(f"Sent: Client: {client_id} Round: {c_round}")

    # Close the connection
    connection.close()


# standardize numerical columns
def standardize(df, col):
    df[col] = (df[col] - df[col].mean()) / df[col].std()


data_st = data.copy()
for i in (data_st.iloc[:, :-1].columns):
    standardize(data_st, i)

data_st.head()

train_data_st = data_st.values
train_data_st

labels = labels_full.values
labels

x_train, x_test, y_train, y_test = train_test_split(train_data_st, labels, test_size=0.2, random_state=42, shuffle=True)
# x_train, x_validate, y_train, y_validate = train_test_split(x_train, y_train, test_size=0.125)

print(x_train.shape)
print(x_test.shape)
# print(x_validate.shape)


x_train_cnn = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test_cnn = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
# x_validate_cnn = np.reshape(x_validate, (x_validate.shape[0], x_validate.shape[1],1))
print(x_train_cnn.shape)
print(x_test_cnn.shape)
# print(x_validate_cnn.shape)

# Load and compile Keras model

model = Sequential()
model.add(Conv1D(filters=64, kernel_size=5, strides=1, padding='same', input_shape=(train_data_st.shape[1], 1)))
# model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding='same'))
model.add(LSTM(32, activation='relu', return_sequences=True))
# model.add(LSTM(16, return_sequences=True))  # returns a sequence of vectors of dimension 16
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

model.add(Dense(labels.shape[1], activation='softmax'))
# model = Sequential()
# model.add(Conv1D(filters=32, kernel_size=5, strides=1, padding='same', input_shape = (train_data_st.shape[1], 1)))
# model.add(LSTM(16, return_sequences=True))  # returns a sequence of vectors of dimension 16
# model.add(Flatten())
# model.add(Dense(8, activation='relu'))

# model.add(Dense(labels.shape[1],activation='softmax'))


modelName = 'CNN_LSTM_Client_1_epoch5_5_'
# keras.utils.plot_model(model, './' + modelName + '_Archi.png', show_shapes=True)
model.summary()

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
# sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                            patience=3,
                                            verbose=1,
                                            factor=0.5,
                                            lr=0.00001)
earlystop = EarlyStopping(monitor='val_loss',
                          min_delta=0,
                          patience=10,
                          verbose=1,
                          restore_best_weights=True)

checkpoint = ModelCheckpoint('./' + modelName + str(random.randint(1, 100)) + '.h5',
                             monitor='val_loss',
                             mode='min',
                             save_best_only=True,
                             save_weights_only=True,
                             verbose=1)


# Define Flower client
class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        model.set_weights(parameters)
        # epochs = 5
        # batch_size = 256
        history = model.fit(x_train_cnn, y_train, batch_size=batch_size,
                            steps_per_epoch=x_train.shape[0] // batch_size,
                            epochs=epochs,
                            # validation_data=(x_validate_cnn,y_validate),
                            validation_data=(x_test_cnn, y_test),
                            # validation_split=0.10,
                            callbacks=[learning_rate_reduction, checkpoint]
                            )
        print("Fit history : ", history)

        def plot_model_history(model_history):
            fig, axs = plt.subplots(1, 2, figsize=(15, 5))
            # summarize history for accuracy
            axs[0].plot(range(1, len(model_history.history['accuracy']) + 1), model_history.history['accuracy'], '--*',
                        color=(1, 0, 0))
            axs[0].plot(range(1, len(model_history.history['val_accuracy']) + 1), model_history.history['val_accuracy'],
                        '-^', color=(0.7, 0, 0.7))
            axs[0].set_title('Model ' + modelName + ' Accuracy')
            axs[0].set_ylabel('Accuracy')
            axs[0].set_xlabel('Epoch')
            # axs[0].set_xticks(np.arange(1,len(model_history.history['accuracy'])+1),len(model_history.history['accuracy'])/[10])
            axs[0].legend(['train', 'val'], loc='best')
            axs[0].grid('on')
            # summarize history for loss
            axs[1].plot(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'], '-x',
                        color=(0, 0.5, 0))
            axs[1].plot(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'], '-.D',
                        color=(0, 0, 0.5))
            axs[1].set_title('Model ' + modelName + ' Loss')
            axs[1].set_ylabel('Loss')
            axs[1].set_xlabel('Epoch')
            # axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
            axs[1].legend(['train', 'val'], loc='best')
            axs[1].grid('on')
            # plt.savefig('./'+modelName+'.jpg',dpi=600)
            # plt.show()

        plot_model_history(history)
        with open('./History_' + modelName, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return model.get_weights(), len(x_train), results

    def evaluate(self, parameters, config):
        global current_round
        current_round += 1
        print("\n\n\n----------------  Test ----------------- ")
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test_cnn, y_test, verbose=1)
        print("Eval accuracy : ", accuracy)
        y_pred = model.predict(x_test)

        y_pred_cm = np.argmax(y_pred, axis=1)
        y_test_cm = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_test_cm, y_pred_cm)

        group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]

        group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]

        labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]

        labels = np.asarray(labels).reshape(11, 11)

        label = ['benign', 'mirai_udp', 'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan', 'gafgyt_tcp', 'gafgyt_udp' \
            , 'mirai_ack', 'mirai_scan', 'mirai_syn', 'mirai_udpplain']

        plt.figure(figsize=(11, 11))
        sns.heatmap(cm, xticklabels=label, yticklabels=label, annot=labels, fmt='', cmap="Blues", vmin=0.2);
        plt.title('Confusion Matrix for' + modelName + ' model')
        plt.ylabel('True Class')
        plt.xlabel('Predicted Class')
        # plt.savefig('./'+modelName+'_CM.png')
        # plt.show()

        cr = classification_report(y_test_cm, y_pred_cm,
                                   target_names=['benign', 'mirai_udp', 'gafgyt_combo', 'gafgyt_junk', 'gafgyt_scan',
                                                 'gafgyt_tcp', 'gafgyt_udp', 'mirai_ack', 'mirai_scan', 'mirai_syn',
                                                 'mirai_udpplain'])
        print(cr)

        # loss, accuracy = model.evaluate(x_test_cnn, y_test, verbose=1)
        print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

        with open('./' + modelName + '_CR.txt', 'a') as f:
            f.write(classification_report(y_test_cm, y_pred_cm,
                                          target_names=['benign', 'mirai_udp', 'gafgyt_combo', 'gafgyt_junk',
                                                        'gafgyt_scan', 'gafgyt_tcp', 'gafgyt_udp', 'mirai_ack',
                                                        'mirai_scan', 'mirai_syn', 'mirai_udpplain']))
            f.write("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss))

        from itertools import cycle
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(labels.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        colors = cycle(['blue', 'red', 'green', 'aqua', 'darkorange', 'orange', 'fuchsia', 'lime', 'magenta'])
        for i, color in zip(range(labels.shape[1]), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                           ''.format(i, roc_auc[i]))
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.ylabel('Recall')
        plt.xlabel('Fall-out (1-Specificity)')
        plt.title('Receiver Operating Characteristic (ROC) for ' + modelName + ' model')
        plt.legend(loc="lower right")
        # plt.savefig('./'+modelName+'_ROC.png')
        #
        # plt.show()
        call_mq(accuracy, loss, current_round, cr)
        return loss, len(x_test_cnn), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address=f"{server_url}:8080", client=FlowerClient()
)
