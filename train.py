import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


import configur
import util


def get_xy(data):
    x = data.drop(configur.target, axis=1)
    y = data[configur.target]
    util.logger.info(f"x shape: {x.shape}, y shape: {y.shape}")

    return x,y

def normalize_input(x, option="minmax"):
    if option=="minmax":
        scaler = MinMaxScaler()
        x = scaler.fit_transform(x)
    if option=="keras":
        layer = tf.keras.layers.Normalization(axis=None)
        layer.adapt(x)
        x = layer(x)
    
    return x

def split_data(x, y):
    n = int(len(x) * configur.train_test_size)
    xtr, xvl = x[:n, :], x[n:, :]
    ytr, yvl = y[:n], y[n:]

    util.logger.info(f"data splited, train/test shapes: {xtr.shape, ytr.shape}/{xvl.shape, yvl.shape}")
    
    return xtr, xvl, ytr, yvl

def preprocess(data):
    x,y = get_xy(data)
    x = normalize_input(x)
    xtr, xts, ytr, yts = split_data(x, y)
    xtr = xtr.reshape(xtr.shape[0], xtr.shape[1], 1)
    xts = xts.reshape(xts.shape[0], xts.shape[1], 1)

    return xtr, xvl, ytr, yvl


def make_convolution_model(input_shape):
    input_layer = keras.layers.Input(shape=input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.ReLU()(conv1)

    conv2 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.ReLU()(conv2)

    conv3 = keras.layers.Conv1D(filters=128, kernel_size=3, padding="same")(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.ReLU()(conv3)

    gap = keras.layers.GlobalAveragePooling1D()(conv3)

    output_layer = keras.layers.Dense(1, activation="relu")(gap)

    return keras.models.Model(inputs=input_layer, outputs=output_layer)

def fit_convolution(data):
    st = time.time()
    util.logger.info(f"start fitting convolution model")

    x,y = get_xy(data)
    x = normalize_input(x)
    xtr, xts, ytr, yts = split_data(x, y)
    xtr = xtr.reshape(xtr.shape[0], xtr.shape[1], 1)
    xts = xts.reshape(xts.shape[0], xts.shape[1], 1)

    model = make_convolution_model(xtr.shape[1:])

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "models/conv_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", patience=20, factor=0.1,min_lr = 0.0001
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20)
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.Huber(delta=1.0),
        metrics=["mean_absolute_error"]
    )

    history = model.fit(
        xtr, ytr,
        batch_size=configur.batch_size,
        epochs = configur.epochs,
        callbacks = callbacks,
        validation_split=0.2,
        verbose=1
    )

    # test
    test_loss, test_acc = model.evaluate(xts, yts)
    util.logger.info(f"test loss: {test_loss}, test acc: {test_acc}")

    # plot hisotry 
    metric = "mean_absolute_error"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()

    util.logger.info(f"end fitting convolution model, time: {time.time() - st}")

    return model, history

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = keras.layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = keras.layers.Dropout(dropout)(x)
    x = keras.layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x)

    return x + res

def make_transformer_model(
    classes, input_shape, head_size, num_heads, ff_dim, num_transformer_blocks,
    mlp_units, dropout=0, mlp_dropout=0
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)
    x = keras.layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = keras.layers.Dense(dim, activation="relu")(x)
        x = keras.layers.Dropout(mlp_dropout)(x)
    outputs = keras.layers.Dense(len(classes), activation="softmax")(x)

    return keras.Model(inputs, outputs)

def fit_transformer(data):
    st = time.time()
    util.logger.info(f"start fitting transformer model")

    x,y = get_xy(data)
    x = normalize_input(x)
    xtr, xts, ytr, yts = split_data(x, y)
    xtr = xtr.reshape(xtr.shape[0], xtr.shape[1], 1)
    xts = xts.reshape(xts.shape[0], xts.shape[1], 1)

    model = make_transformer_model(
        classes=list(set(ytr)),
        input_shape = xtr.shape[1:],
        head_size=256,
        num_heads=4,
        ff_dim=4,
        num_transformer_blocks=4,
        mlp_units=[128],
        mlp_dropout=0.4,
        dropout=0.25,
    )

    callbacks = [
        keras.callbacks.ModelCheckpoint(
            "models/transform_model.h5", save_best_only=True, monitor="val_loss"
        ),
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, verbose=1)
    ]

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        # loss=keras.losses.Huber(delta=1.0),
        # metrics=["mean_absolute_error"],
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"]
    )

    history = model.fit(
        xtr, ytr,
        batch_size=configur.batch_size,
        epochs = configur.epochs,
        callbacks = callbacks,
        validation_split=0.2,
        verbose=1
    )

    # test
    test_loss, test_acc = model.evaluate(xts, yts)
    util.logger.info(f"test loss: {test_loss}, test acc: {test_acc}")

    # plot hisotry 
    metric = "sparse_categorical_accuracy"
    plt.figure()
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric])
    plt.title("model " + metric)
    plt.ylabel(metric, fontsize="large")
    plt.xlabel("epoch", fontsize="large")
    plt.legend(["train", "val"], loc="best")
    plt.show()

    util.logger.info(f"end fitting transformer model, time: {time.time() - st}")

    return model, history