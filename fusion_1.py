
import pandas as pd

from tensorflow.keras.layers import Input, Conv2D, Concatenate, MaxPool2D, \
    Dropout, Flatten, Dense, LSTM, Reshape, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from generate_data import data_generator


maxlen = 200

def face_model():
    input_ = Input(shape=(224,224,3), name='face_input')
    x = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(input_)
    x = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), name='face_vgg16')(x)
    x = Dropout(0.5)(x)
    x = Flatten(name='face_flatten')(x)
    # x = Dense(256, activation='relu', name='face_fc1')(x)
    # x = Dense(128, activation='relu', name='face_fc2')(x)
    # x = Dense(20, activation='softmax', name='face_output')(x)
    model = Model(inputs=input_, outputs=x)
    return model


def palm_print_model():
    input_ = Input(shape=(224,224,3), name='palmprint_input')
    x = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(input_)
    x = Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
    x = MaxPool2D(pool_size=(2,2), strides=(2,2), name='palm_vgg16')(x)
    x = Dropout(0.5)(x)
    x = Flatten(name='palm_flatten')(x)
    # x = Dense(256, activation='relu', name='palm_fc1')(x)
    # x = Dense(128, activation='relu', name='palm_fc2')(x)
    # x = Dense(20, activation='softmax', name='palm_output')(x)
    model = Model(inputs=input_, outputs=x)
    return model


def signature_model():
    input_ = Input(shape=(224, 224, 3), name='signature_input')
    x = Conv2D(32, 3, activation='relu', padding='same')(input_)
    x = MaxPool2D(2, padding='same')(x)
    x = Conv2D(64, 3, activation='relu', padding='same')(x)
    x = MaxPool2D(2, padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPool2D(2, padding='same')(x)
    x = Conv2D(128, 3, activation='relu', padding='same')(x)
    x = MaxPool2D(2, padding='same')(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    # x = Dense(20, activation='softmax', name='signature_output')(x)
    model = Model(inputs=input_, outputs=x)
    return model


def signature_model2():
    input_ = Input(shape=(maxlen, 7), name='signature_input')
    x = LSTM(128, return_sequences=True)(input_)
    x = LSTM(64)(x)
    x = Flatten()(x)
    # x = Dense(20, activation='softmax',name='signature_output')(x)
    return Model(inputs=input_, outputs=x)


def audio_model():
    input_shape = (99, 13, 1)
    input_ = Input(shape=input_shape, name='audio_input')
    x = Conv2D(32, 3, activation='relu')(input_)
    x = Conv2D(8, 3, activation='relu')(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    x = Conv2D(32, 3, activation='relu')(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    x = Conv2D(64, 2, activation='relu')(x)
    x = MaxPool2D(3, strides=2, padding='same')(x)
    x = Reshape((-1, 64))(x)
    # Structural Feature Extraction from LSTM
    x = LSTM(64, return_sequences=True)(x)
    x = LSTM(64)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    # x = Dense(20, activation='softmax', name='audio_output')(x)
    model = Model(inputs=input_, outputs=x)
    return model


def audio_model_cnn():
    input_shape = (99, 13, 1)
    input_ = Input(shape=input_shape, name='audio_input')
    x = Conv2D(16, 3, activation='relu', strides=1, padding='same')(input_)
    x = Conv2D(32, 3, activation='relu', strides=1, padding='same')(x)
    x = Conv2D(64, 3, activation='relu', strides=1, padding='same')(x)
    x = Conv2D(128, 3, activation='relu', strides=1, padding='same')(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(64, activation='relu')(x)
    # x = Dense(20, activation='softmax', name='audio_output')(x)
    model = Model(inputs=input_, outputs=x)
    return model


def audio_model_lstm():
    input_shape = (99, 13)
    input_ = Input(shape=input_shape, name='audio_input')
    x = LSTM(128, return_sequences=True)(input_)
    x = LSTM(128, return_sequences=True)(x)
    x = Dropout(0.5)(x)
    x = TimeDistributed(Dense(64, activation='relu'))(x)
    x = TimeDistributed(Dense(32, activation='relu'))(x)
    x = TimeDistributed(Dense(16, activation='relu'))(x)
    x = TimeDistributed(Dense(8, activation='relu'))(x)
    x = Flatten()(x)
    # x = Dense(20, activation='softmax', name='audio_output')(x)
    model = Model(inputs=input_, outputs=x)
    return model


def model():
    f_model = face_model()
    p_model = palm_print_model()
    # s_model = signature_model()
    s_model = signature_model2()
    # a_model = audio_model()
    a_model = audio_model_cnn()

    merge_1 = Concatenate(axis=1)([s_model.output, a_model.output])
    # merge_1 = BatchNormalization()(merge_1)
    merge_1 = Dense(128, activation='relu')(merge_1)
    # merge_1 = Dense(64, activation='relu')(merge_1)

    merge_2 = Concatenate(axis=1)([p_model.output, merge_1])
    # merge_2 = BatchNormalization()(merge_2)
    merge_2 = Dense(64, activation='relu')(merge_2)
    # merge_2 = Dense(32, activation='relu')(merge_2)

    # merge_3  = Concatenate(axis=1)([f_model.output, merge_2])

    merge_3 = f_model.output
    merge_3 = Dense(256, activation='relu')(merge_3)
    merge_3 = Dense(128, activation='relu')(merge_3)
    # merge_3 = BatchNormalization()(merge_3)
    # merge_3 = Dense(32, activation='relu')(merge_3)
    merge_3 = Dense(20, activation='softmax')(merge_3)

    final_cas_model = Model(inputs=[f_model.input, p_model.input, s_model.input,
                                    a_model.input],
                            outputs=[merge_3])
    final_cas_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy',
                            metrics=['accuracy'])

    train_batch_size = 8
    val_batch_size = 4
    num_train_samples = pd.read_csv('train.csv').shape[0]
    num_val_samples = pd.read_csv('val.csv').shape[0]
    steps_per_epoch = num_train_samples//train_batch_size
    validation_steps = num_val_samples//val_batch_size

    train_gen = data_generator('train.csv', batch_size=train_batch_size)
    val_gen = data_generator('val.csv', batch_size=val_batch_size)
    history = final_cas_model.fit(train_gen, epochs=25,
                                  steps_per_epoch=steps_per_epoch,
                                  validation_data=val_gen,
                                  validation_steps=validation_steps)


if __name__ == "__main__":
    model()
