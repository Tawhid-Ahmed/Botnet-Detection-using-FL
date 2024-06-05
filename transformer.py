# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 12:50:55 2023

@author: tawhid
"""
import argparse

# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 09:06:01 2023

@author: tawhid
"""

"""python3 transformer.py -l from which nummber of row to start skip 
-t to which nummber of row to skip 
-c Range of row to count 
-n dataset_number 
-client_id client_id_number"""

"""python3 transformer.py -l 1 -t 100 -c 1000 -n 4 -client_id """

import requests
import flwr as fl
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

from keras.layers import Conv1D, MaxPool1D, BatchNormalization, Activation, Input, Add, \
    GlobalAveragePooling1D, Dense, concatenate
from keras.models import Model
# from tensorflow.keras.optimizers import Adam, RMSprop
# from tensorflow.keras.metrics import Recall, Precision
import keras
import time
import os
import pika
import json

# parameter_value = os.environ.get('PARAMETER_NAME')

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.8  # Adjust this value as needed
# session = tf.compat.v1.Session(config=config)
#
# tf.profiler.experimental.start('logs/memory_profiler')

# skip=0
# count=100

# argParser = argparse.ArgumentParser()
# argParser.add_argument("-l", "--low", default=0, type=int, help="from which nummber of row to start skip")
# argParser.add_argument("-t", "--top", default=0, type=int, help="to which nummber of row to skip")
# argParser.add_argument("-c", "--count", default=1000, type=int, help="Range of row")
# argParser.add_argument("-n", "--num", default=2, type=int, help="dataset_number")
# argParser.add_argument("-ci", "--client_id", default=1, type=int, help="client_id_number")
#
# args = argParser.parse_args()
# # print("args=%s" % args)
#
# # print("args.name=%s" % args.name)
#
#
# low = args.low
# top = args.top
# skip = [i for i in range(low, top)]  # args.skip
# count = args.count
# client_id = args.client_id
# dataset_num = args.num

def custom_exception_handler(exc_type, exc_value, exc_traceback):
    # Customize error handling here
    print("An unhandled exception occurred:")
    print(f"Type: {exc_type}")
    print(f"Value: {exc_value}")
    # You can log the error or perform any other action here
    input("There are errors. Press Enter to continue...")


# Set the custom exception handler
# sys.excepthook = custom_exception_handler

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

# count = 1000
# data = pd.concat([benign_df.head(count), m_u_df.head(count), g_c_df.head(count),
#                 g_j_df.head(count), g_s_df.head(count), g_t_df.head(count),
#                 g_u_df.head(count), m_a_df.head(count), m_sc_df.head(count),
#                 m_sy_df.head(count), m_u_p_df.head(count)],
#                axis=0, sort=False, ignore_index=True)

labels_full = pd.get_dummies(data['type'], prefix='type')
labels_full.head()

data = data.drop(columns='type')
data.head()


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
X_train, X_test, y_train, y_test = train_test_split(train_data_st, labels, test_size=0.2, random_state=42, shuffle=True)

# train_df, test_df = train_test_split(data, test_size=0.2, random_state=42, stratify=data["type"])


# features = list(train_df.columns)
# features.remove("type")


# from sklearn.preprocessing import LabelEncoder

# label_encoder = LabelEncoder()
# train_df["type"] = label_encoder.fit_transform(train_df["type"])
# test_df["type"] = label_encoder.transform(test_df["type"])

# train_df = pd.get_dummies(train_df, columns=["type"])
# test_df = pd.get_dummies(test_df, columns=["type"])


# from sklearn.preprocessing import MinMaxScaler

# scaler = MinMaxScaler()
# train_df[features] = scaler.fit_transform(train_df[features])
# test_df[features] = scaler.transform(test_df[features])

# X_train = train_df[features].values
# y_train = train_df["type"].values

# X_test = test_df[features].values
# y_test = test_df["type"].values


# from sklearn.ensemble import ExtraTreesClassifier
# from sklearn.feature_selection import SelectFromModel


# clf = ExtraTreesClassifier(n_estimators=50, n_jobs=-1)
# clf = clf.fit(X_train, y_train)
# clf.feature_importances_  


# ext=pd.DataFrame(clf.feature_importances_,columns=["extratrees_importance"])
# ext = ext.sort_values(['extratrees_importance'], ascending=False)
# feature_index = [features[i] for i in list(ext.index)]
# ext["Feature_Name"] = feature_index
# ext

# model = SelectFromModel(clf, prefit=True)
# X_train = model.transform(X_train)
# X_test = model.transform(X_test)
# X_test.shape 


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))  # X_train.reshape((-1, X_train.shape[-1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))  # X_test.reshape((-1, X_test.shape[-1], 1))

# test_batched = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(len(y_test))

# X_train, x_validate, y_train, y_validate = train_test_split(X_train, y_train, test_size=0.125)

from sklearn.utils import class_weight

# class_weights = class_weight.compute_class_weight('balanced',
#                                                  classes=np.unique(y_train),
#                                                  y=y_train)

# class_weights = {k: v for k,v in enumerate(class_weights)}
# class_weights


input_shape = X_train.shape[1:]
nb_classes = 11
print(nb_classes)

# from tensorflow.keras.utils import to_categorical

# def convert_to_categorical(y, nb_classes):
#     return to_categorical(y, num_classes=nb_classes)

# y_test = convert_to_categorical(y_test, nb_classes)

from keras import layers
from keras.layers import Conv1D, MaxPooling1D, Concatenate, Activation, BatchNormalization


class INCEPTION_Block(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(INCEPTION_Block, self).__init__(**kwargs)
        self.kernel_size = 50
        #         f1, f2_in, f2_out, f3_in, f3_out, f4_out = (128, 128, 192, 32, 96, 64)
        f1, f2_in, f2_out, f3_in, f3_out, f4_out = (16, 16, 24, 4, 12, 8)
        kernel_size_s = [10, 30, 50]

        # 1x1 conv
        self.conv_1_1 = Conv1D(f1, kernel_size_s[0], padding='same', activation='relu')

        # 3x3 conv
        self.conv_1_2 = Conv1D(f2_in, kernel_size_s[0], padding='same', activation='relu')
        self.conv_3_2 = Conv1D(f2_out, kernel_size_s[1], padding='same', activation='relu')

        # 5x5 conv
        self.conv_1_3 = Conv1D(f3_in, kernel_size_s[0], padding='same', activation='relu')
        self.conv_5_3 = Conv1D(f3_out, kernel_size_s[2], padding='same', activation='relu')

        # 3x3 max pooling
        self.pool = MaxPooling1D(kernel_size_s[1], strides=1, padding='same')
        self.conv_final = Conv1D(f4_out, kernel_size_s[0], padding='same', activation='relu')

        # concatenate filters, assumes filters/channels last
        self.concatenate = Concatenate(axis=-1)

        self.batch_normalization = BatchNormalization()
        self.relu_activation = Activation(activation='relu')

    def call(self, layer_in):
        # 1x1 conv
        conv1 = self.conv_1_1(layer_in)

        # 3x3 conv
        conv3 = self.conv_1_2(layer_in)
        conv3 = self.conv_3_2(conv3)

        # 5x5 conv
        conv5 = self.conv_1_3(layer_in)
        conv5 = self.conv_5_3(conv5)

        # 3x3 max pooling
        pool = self.pool(layer_in)
        pool = self.conv_final(pool)

        # concatenate filters, assumes filters/channels last
        layer_out = self.concatenate([conv1, conv3, conv5, pool])

        layer_out = self.batch_normalization(layer_out)
        layer_out = self.relu_activation(layer_out)

        return layer_out


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

def call_api(ac, ls, c_round, cr):
    url = f'http://{server_url}:5000/model_data'  # Change the URL if your server is running on a different address/port

    res = {
        'client_id': client_id,
        'accuracy': ac,
        'loss': ls,
        'round': c_round,
        'cr': cr
    }

    response = requests.post(url, json=res)

    if response.status_code == 200:
        print('API Response:', response.json())
    else:
        print('API Error:', response.status_code, response.text)


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Attention and Normalization
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    # Feed Forward Part
    x = keras.models.Sequential(
        [INCEPTION_Block(),
         INCEPTION_Block(),
         INCEPTION_Block()])(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res


class TransformerEncoder(layers.Layer):
    def __init__(self, head_size, num_heads, neurons):
        super(TransformerEncoder, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.ffn = keras.models.Sequential(
            # [layers.Dense(neurons, activation="relu"), layers.Dense(head_size),]
            [
                INCEPTION_Block(),
                INCEPTION_Block(),
                INCEPTION_Block(),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


def get_model(input_shape, nb_classes) -> tf.keras.Model:
    head_size = 64  # Embedding size for attention
    num_heads = 3  # Number of attention heads
    ff_dim = 128  # Hidden layer size in feed forward network inside transformer
    num_transformer_blocks = 1
    mlp_units = [32]
    mlp_dropout = 0.1
    dropout = 0.1

    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(nb_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs)


from keras.metrics import Recall, Precision

# from keras.utils.vis_utils import plot_model

learning_rate = 1e-2
comms_round = 10
loss = 'categorical_crossentropy'
metrics = ["accuracy"]
optimizer = tf.keras.optimizers.legacy.RMSprop(learning_rate=learning_rate)
adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# initialize global model
local_model = get_model(input_shape, nb_classes)
modelName = 'Transformer_{client_id}'.format(client_id=client_id)
# keras.utils.plot_model(local_model, './'+modelName+'_tawhid.png',show_shapes=True)

local_model.summary()

local_model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
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
        return [np.random.rand(*w.shape) for w in local_model.get_weights()]

    def fit(self, parameters, config):
        print("\n\n\n----------------  Train ----------------- ")
        parameters = local_model.get_weights()
        local_model.set_weights(parameters)
        # epochs = 5
        # batch_size = 5
        history = local_model.fit(X_train, y_train, batch_size=batch_size,
                                  steps_per_epoch=X_train.shape[0] // batch_size,
                                  epochs=epochs,
                                  # validation_data=(x_validate,y_validate),
                                  validation_split=0.10,
                                  callbacks=[learning_rate_reduction, checkpoint]
                                  )

        with open('./History_' + modelName, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)

        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        return local_model.get_weights(), len(X_train), results

    def evaluate(self, parameters, config):
        global current_round
        current_round += 1

        print("\n\n\n----------------  Test ----------------- ")
        parameters = local_model.get_weights()
        local_model.set_weights(parameters)
        # loss, accuracy = local_model.evaluate(X_test , y_test, verbose=1)
        # print("Eval accuracy : ", accuracy)

        # del X_test, y_test

        from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_curve, auc, \
            classification_report
        import tensorflow as tf
        # model = tf.keras.models.load_model('/'+model_name+'.h5')

        y_pred = local_model.predict(X_test)

        y_pred_cm = np.argmax(y_pred, axis=1)
        y_test_cm = np.argmax(y_test, axis=1)

        cm = confusion_matrix(y_test_cm, y_pred_cm)

        cr = classification_report(y_test_cm, y_pred_cm, target_names= ['benign','mirai_udp','gafgyt_combo','gafgyt_junk','gafgyt_scan','gafgyt_tcp','gafgyt_udp','mirai_ack','mirai_scan','mirai_syn','mirai_udpplain'])
        print(cr)

        loss_val, accuracy = local_model.evaluate(X_test, y_test, verbose=1)
        print("Test: accuracy = %f  ;  loss = %f" % (accuracy, loss_val))
        # cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        # loss = cce(y_test, y_pred)
        # print("loss: ")
        # print(loss)
        # y_hat = np.argmax(y_pred, axis=1)
        # y_true = np.argmax(y_test, axis=1)

        # accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))
        # print(accuracy)
        call_mq(accuracy, loss_val, current_round, cr)
        return loss_val, len(X_test), {"accuracy": accuracy}


# Start Flower client
fl.client.start_numpy_client(
    server_address=f"{server_url}:8080", client=FlowerClient()
)
