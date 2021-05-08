import os
import pandas as pd
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.layers import Dense,Flatten,Conv2D,Concatenate,RepeatVector,Dropout,Input,MaxPooling2D,Conv1D,Lambda,MaxPool1D
from keras.models import Sequential, Model
from keras.layers import LSTM,Reshape,BatchNormalization
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers import MaxPool2D
import tensorflow as tf
from sklearn.model_selection import train_test_split
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank, delta
from tqdm import tqdm
import pandas as pd
import matplotlib as plt
import keras
from keras.utils import to_categorical

def read_data(main_dir):


    folder_name = os.listdir(main_dir)
    persons =[]

    for person_name in (folder_name):
        folder_path = os.listdir(main_dir + '/' + person_name)
        count = 0
        for filename in folder_path:
            count += 1
            persons.append(((person_name, str(main_dir + '/' + person_name) + '/' + filename, person_name + '_' + str(count))))


        person_df = pd.DataFrame(data=persons, columns=['name',main_dir,'id_index'])
        person_df.to_csv(main_dir+".csv")

#main function
# read_data('face')
# read_data('palmprint')
# read_data('signature_data')
#read_data('speaker')

merge_fpsa_df = pd.read_csv('merge_fpsa.csv')
x_face_path = merge_fpsa_df['face'].tolist()
x_palm_path = merge_fpsa_df['palmprint'].tolist()
x_signature_path = merge_fpsa_df['signature_data'].tolist()
x_audio = merge_fpsa_df['speaker'].tolist()
y_id = merge_fpsa_df['name_x'].tolist()


# face_imgs =[]
# palm_imgs=[]
# signature_imgs=[]
# id_labels =[]
im_size = 224
# images =[]
# labels =[]

def path2image(path_list):
    append_list = []
    for path in path_list:
        img = cv2.imread(path)
        img = cv2.resize(img,(im_size,im_size))
        append_list.append(img)
    return append_list



face_imgs = path2image(x_face_path)
face_imgs = np.asarray(face_imgs)

palm_imgs = path2image(x_palm_path)
palm_imgs = np.asarray(palm_imgs)

signature_imgs = path2image(x_signature_path)
signature_imgs = np.asarray(signature_imgs)

y_labelencod = LabelEncoder()
y = y_labelencod.fit_transform(y_id)
y = y.reshape(-1,1)

onehot = OneHotEncoder(categories='auto')

Y = onehot.fit_transform(y)
print(y_id,y,Y)



df = pd.read_csv('audio.csv')
df.set_index('audio', inplace=True)
for f in df.index:
    rate,signal = wavfile.read('clean/'+f)
    df.at[f,'length']  =signal.shape[0]/rate

classes = list(np.unique(df.name))
class_dist = df.groupby(['name'])['length'].mean()

n_samples = 2*int(df['length'].sum()/0.1)
prob_dist = class_dist/class_dist.sum()
choices = np.random.choice(class_dist.index, p=prob_dist)

def build_rand_feat():
    X = []
    y =[]
    _min, _max = float('inf'), -float('inf')
    for _ in tqdm(range(n_samples)):
        rand_class = np.random.choice(class_dist.index,p=prob_dist)
        file = np.random.choice(df[df.name==rand_class].index)
        rate, wav = wavfile.read('clean/'+file)
        label = df.at[file,'name']
        rand_index = np.random.randint(0,wav.shape[0]-config.step)
        sample = wav[rand_index:rand_index+config.step]
        X_sample = mfcc(sample,rate,numcep=config.nfeat, nfilt=config.nfilt, nfft=config.nfft).T
        _min = min(np.amin(X_sample), _min)
        _max = max(np.amax(X_sample), _max)
        X.append(X_sample if config.mode =='conv' else X_sample.T)
        y.append(classes.index(label))
    X,y = np.array(X), np.array(y)
    X = (X - _min)/(_max - _min)
    if config.mode == 'conv':
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
        print(X.shape)
    elif config.mode == 'time':
        X = X.reshape(X.shape[0],X.shape[1],X.shape[2])
    y = to_categorical(y, num_classes=20)
    return X,y




#face
x1 = Input(shape=(224,224,3),name='face_input')
x = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(x1)
x = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
x = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
x = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
x = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2))(x)
x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
x = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(x)
x = MaxPool2D(pool_size=(2,2),strides=(2,2),name='face_vgg16')(x)
x = Dropout(0.5)(x)
x = Flatten(name='face_flatten')(x)
x = Dense(256, activation='relu', name='face_fc1')(x)
x = Dense(128, activation='relu', name='face_fc2')(x)
out1 = Dense(20, activation='softmax',name='face_output')(x)


#palmprint
x2 = Input(shape=(224,224,3), name='palmprint_input')
xp = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(x2)
xp = Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu")(xp)
xp = MaxPool2D(pool_size=(2,2),strides=(2,2))(xp)
xp = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = MaxPool2D(pool_size=(2,2),strides=(2,2))(xp)
xp = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = MaxPool2D(pool_size=(2,2),strides=(2,2))(xp)
xp = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = MaxPool2D(pool_size=(2,2),strides=(2,2))(xp)
xp = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu")(xp)
xp = MaxPool2D(pool_size=(2,2),strides=(2,2),name='palm_vgg16')(xp)
xp = Dropout(0.5)(xp)
xp = Flatten(name='palm_flatten')(xp)
xp = Dense(256, activation='relu', name='palm_fc1')(xp)
xp = Dense(128, activation='relu', name='palm_fc2')(xp)
out2 = Dense(20, activation='softmax', name='palm_output')(xp)

#signature
x3 = Input(shape=(224,224,3), name='signature_input')
#cnn.add(Reshape((1,train_x.shape[1],1)))
xs = Conv2D(32,3, activation='relu',padding='same')(x3)
xs = MaxPooling2D(2,padding='same')(xs)
xs = Conv2D(64,3, activation='relu',padding='same')(xs)
xs = MaxPooling2D(2,padding='same')(xs)
xs =Conv2D(128, 3, activation='relu',padding='same')(xs)
xs=MaxPooling2D(2,padding='same')(xs)
xs=Conv2D(128, 3, activation='relu',padding='same')(xs)
xs=MaxPooling2D(2,padding='same')(xs)
xs = Dropout(0.5)(xs)
xs=Flatten()(xs)
out3=Dense(20, activation='softmax', name='signature_output')(xs)

#audio
x4 = Input(shape=input_shape, name = 'spaker_input')
xa = Conv2D(32, 3,activation='relu')(x4)
xa = Conv2D(8, 3, activation='relu')(xa)
xa = MaxPooling2D(3, strides=2, padding='same')(xa)
xa=Conv2D(32, 3, activation='relu')(xa)
xa=MaxPooling2D(3, strides=2, padding='same')(xa)
xa=Conv2D(64, 2, activation='relu')(xa)
xa=MaxPooling2D(3, strides=2, padding='same')(xa)
xa=Reshape(-1, 64)

    # Structural Feature Extraction from LSTM
xa=LSTM(64, return_sequences=True)(xa)
xa=LSTM(64)(xa)
xa=BatchNormalization()(xa)
xa=Dropout(0.2)(xa)
out4=Dense(20, activation='softmax')(xa)

#fusion
merge = Concatenate(axis=1)([out1, out2, out3])  
fully = Dense(20, activation='sigmoid')(merge)
fully = Dropout(0.5)(fully)
fully = Dense(20, activation='sigmoid')(fully)
fully = Dropout(0.5)(fully)
fully = Dense(20, activation='sigmoid')(fully)
fully = Dropout(0.5)(fully)
fully = BatchNormalization()(fully)
fc_output = Dense(20, activation='softmax', name='fc_output')(fully)

final_fc_model = Model(inputs=[x1, x2, x3], outputs=[fc_output])
final_fc_model.summary()
#
# callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"),]
final_fc_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
history=final_fc_model.fit({"face_input": face_imgs, "palmprint_input": palm_imgs, "signature_input": signature_imgs, "speaker_input": },Y
                            ,epochs=50,batch_size=20,shuffle=True, validation_split=0.2)





