import cv2
import tensorflow as tf
import numpy as np
import pandas as pd

from argparse import ArugmentParser
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16


image_size = (224, 224)
num_classes = 300
lr = 0.0001
epochs = 5
batch_size = 32


def face_model():
    model = VGG16(include_top=False, input_shape=(224, 224, 3))
    x = Flatten()(model.output)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(4096, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(num_classes, activation='softmax')(x)
    return Model(model.inputs, x)


def read_face_img(image_file):
    image = cv2.imread(image_file)
    image = cv2.resize(image, image_size)
    image = image * 1./255
    return image.astype(np.float32)


def generate_data(csv_file):
    def process():
        df = pd.read_csv(csv_file)
        df = df[['face', 'label']]
        df.drop_duplicates(inplace=True)
        unique_labels = list(np.unique(df.label))
        num_labels = len(unique_labels)
        total_rows = df.shape[0]
        row_idx = 0
        while True:
            if row_idx % total_rows == 0:
                row_idx = 0
                df.sample(frac=1)  # Shuffle dataframe
            row = df.iloc[row_idx, :]
            row_idx += 1
            face = read_face_img(row['face'])
            label = to_categorical(unique_labels.index(row['label']), num_classes=num_labels)
            yield face, label
    return process


def data_generator(csv_file, batch_size):
    gen = generate_data(csv_file)
    dataset = tf.data.Dataset.from_generator(gen,
        output_types=(tf.float32, tf.float32),
        output_shapes=((224, 224, 3), num_classes)
    )
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(2)
    return dataset


if __name__ == "__main__":
	parser = ArugmentParser()
	parser.add_argument("-b", "--batch-size", default=batch_size)
	parser.add_argument("-l", "--learning-rate", default=lr)
	parser.add_argument("-e", "--epochs", default=epochs)
	args = parser.parse_args()

	df = pd.read_csv('train.csv')
	df = df[['face', 'label']]
	df.drop_duplicates(inplace=True)
	num_train_samples = df.drop_duplicates().shape[0]

	df = pd.read_csv('val.csv')
	df = df[['face', 'label']]
	df.drop_duplicates(inplace=True)
	num_val_samples = df.drop_duplicates().shape[0]

	steps_per_epoch = num_train_samples//batch_size
	save_model_as = "face_epochs{}_lr{}_batch{}"
	model_output = save_model_as.format(epochs, lr, batch_size)
	train_gen = data_generator('train.csv', batch_size=batch_size)
	model = face_model()
	model.compile(optimizer=Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

	history = model.fit(train_gen, epochs=epochs,
	                    steps_per_epoch=steps_per_epoch,
	                    validation_data=val_gen,
	                    validation_steps=validation_steps)
	model.save(model_output)

