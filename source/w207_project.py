#!/usr/bin/env python
# coding: utf-8

# # Facial Keypoint Detection
# 


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
import tensorflow
from keras.backend.tensorflow_backend import set_session
from keras.backend.tensorflow_backend import clear_session
from keras.backend.tensorflow_backend import get_session
from tensorflow.keras.layers import LeakyReLU, ReLU
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Activation, Convolution2D, BatchNormalization, Flatten
from tensorflow.keras.layers import Dense, Dropout, Input, MaxPool2D
from tensorflow.keras.optimizers import Nadam
from keras import backend
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import cv2
import os, gc, json, math


# Globals
model_dir = None
history_dir = None
train_file = None
test_file = None
lookid_dir = None
submission_file = None


fill_na = False
add_flip_horiz = True
add_blur_img = False
add_rotate_img = False



def convert_pixels(data):
    """
    Convert pixels to the right intensity 0-1 and in a square matrix.
    """
    data = np.array([row.split(' ') for row in data['Image']],dtype='float') / 255.0
    data = data.reshape(-1,96,96,1)
    return(data)



# Reset Keras Session
def reset_keras():
    sess = get_session()
    clear_session()
    sess.close()
    sess = get_session()

    print(gc.collect()) # if it's done something you should see a number being outputted

    # use the same config as you used to create the session
    config = tensorflow.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 1
    config.gpu_options.visible_device_list = "0"
    set_session(tensorflow.Session(config=config))
    
    return   



def set_file_names():
    
    global train_file, test_file, model_dir, history_dir, lookid_dir, submission_file
    #Create folders if not exist
    if not os.path.exists('../output/'):
        os.makedirs('../output/model')
        os.makedirs('../output/history')
    
    
    model_dir = "../output/model/"
    history_dir = "../output/history/"

    train_file = '../input/training/training.csv'
    test_file = '../input/test/test.csv'
    lookid_dir = '../input/IdLookupTable.csv'
    submission_file = '../output/w207_base_submission.csv'
    
    return



bad_samples = [1747, 1731, 1877, 1881, 1907, 1979, 2090, 2175, 2818, 2562, 2199, 2289, 2321, 2372, 2453, 2505, 2787,
               2811, 2912, 3150, 3173, 3296, 3447, 4180, 4263, 4482, 4490, 4636, 5059, 6493, 6585, 6859, 6808, 6906]



bad_bottom_lip = [210, 350, 499, 512, 810, 839, 895, 1058, 1194,1230, 1245, 1546, 1548]

def fill_mouth_coord(y_train):
    

    for sample in bad_bottom_lip:
        y_train[sample][29] = 94
        y_train[sample][28] = y_train[sample][26]
   
    for sample in range(len(y_train)):
        if(y_train[sample][29] < y_train[sample][27]+1):
             y_train[sample][27] = y_train[sample][29] -1
   
        if((y_train[sample][23] + y_train[sample][25]) > (2*y_train[sample][29]-1)):
             diff = y_train[sample][23] - y_train[sample][25]
       
             if(diff > 0):
                y_train[sample][23] = y_train[sample][29] -1
                y_train[sample][25] = y_train[sample][29] -1 - diff
           
             else:
                y_train[sample][23] = y_train[sample][29] -1 + diff
                y_train[sample][25] = y_train[sample][29] -1 
                
    return(y_train)           



def mean_imputer(y_train):
    # https://stackoverflow.com/questions/18689235/numpy-array-replace-nan-values-with-average-of-columns
    # get column means
    col_mean = np.nanmean(y_train,axis=0)

    # find the x,y indices that are missing from y_train
    inds = np.where(np.isnan(y_train))

    # fill in missing values in y_train with the column means. "take" is much more efficient than fancy indexing
    y_train[inds] = np.take(col_mean, inds[1])
    return


def flip_img_horiz(train_data):
    """
    Flip images horizontally for all training images
    """
    # Flip images
    x_train = convert_pixels(train_data)
    flip_img = np.array([np.fliplr(x_train[[ind]][0]) for ind in range(x_train.shape[0])])
    
    # Flip coordinates
    train_data_flip = train_data.copy()
    x_columns = [col for col in train_data.columns if '_x' in col]
    train_data_flip[x_columns] = train_data[x_columns].applymap(lambda x: 96-x)
    
    #left and right are swapped so undo
    left_columns = [col for col in train_data.columns if 'left' in col]
    right_columns = [col for col in train_data.columns if 'right' in col]
    train_data_flip[left_columns+right_columns] = train_data_flip[right_columns+left_columns]
    
    flip_coord = train_data_flip[[col for col in train_data if col != 'Image']].to_numpy()
    return(flip_img,flip_coord)


    

def blur_img(x_train):
    """
    Add Gaussian blurring to the images
    """
    # https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_filtering/py_filtering.html
    blur_img = np.array([cv2.GaussianBlur(x_train[[ind]][0],(5,5),2).reshape(96,96,1) for ind in range(x_train.shape[0])])
    
    return(blur_img)


def rotate_img(x_train, y_train):
    """"
    Rotate images by angles between [5, 10, 14 degrees]
    """
    angles = [5, -5, 10, -10, 14, -14]
    b = np.ones((1,3))
    rows,cols = (96,96)
    x_train_rot = []
    y_train_rot = y_train.copy()
    M_angles = [cv2.getRotationMatrix2D((cols/2,rows/2),angle,1) for angle in angles]
    
    for i in range(x_train.shape[0]):
        #M = cv2.getRotationMatrix2D((cols/2,rows/2),np.random.choice(angles,1),1)
        M = M_angles[np.random.choice(len(M_angles))]
        x_train_rot.append((cv2.warpAffine(x_train[[i]].reshape(rows,cols,1),M,(cols,rows)).reshape(96,96,1)))
       
        #apply affine transformation to (x,y) labels
        for j in range(int(y_train.shape[1]/2)):
            b[:,0:2] = y_train[i,2*j:2*j+2]
            y_train_rot[i,2*j:2*j+2] = np.dot(b,M.transpose()) 
    
    x_train_rot = np.array(x_train_rot)
    return x_train_rot, y_train_rot




# Define callback function if detailed log required
class History(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.train_loss = []
        self.train_rmse = []
        self.val_rmse = []
        self.val_loss = []

    def on_batch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.train_rmse.append(logs.get('rmse'))
        
    def on_epoch_end(self, batch, logs={}):    
        self.val_rmse.append(logs.get('val_rmse'))
        self.val_loss.append(logs.get('val_loss'))
        
# Implement ModelCheckPoint callback function to save CNN model
class CNN_ModelCheckpoint(tensorflow.keras.callbacks.Callback):

    def __init__(self, model, filename):
        self.filename = filename
        self.cnn_model = model

    def on_train_begin(self, logs={}):
        self.max_val_rmse = math.inf
        
 
    def on_epoch_end(self, batch, logs={}):    
        val_rmse = logs.get('val_rmse')
        if(val_rmse < self.max_val_rmse):
           self.max_val_rmse = val_rmse
           self.cnn_model.save_weights(self.filename)


def base_model():
    model_input = Input(shape=(96,96,1))

    x = Convolution2D(32, (3,3), padding='same', use_bias=False)(model_input)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(32, (3,3), padding='same', use_bias=False)(model_input)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Convolution2D(64, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(64, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Convolution2D(96, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(96, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Convolution2D(128, (3,3),padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(128, (3,3),padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.1)(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Convolution2D(256, (3,3),padding='same',use_bias=False)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(256, (3,3),padding='same',use_bias=False)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(0.1)(x)
    x = Convolution2D(512, (3,3), padding='same', use_bias=False)(x)
    x = LeakyReLU(alpha = 0.1)(x)
    x = BatchNormalization()(x)
    x = Convolution2D(512, (3,3), activation='relu', padding='same', use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512,activation='relu')(x)
    x = Dropout(0.1)(x)
    model_output = Dense(30)(x)
    model = Model(model_input, model_output, name="base_model")
    return model




# In[18]:





# Custom RMSE metric
def rmse(y_true, y_pred):
    return backend.sqrt(backend.mean(backend.square(y_pred - y_true), axis=-1))

#from tensorflow.keras.optimizers.schedules import InverseTimeDecay

def model_training(x_train, y_train):
    
    model = base_model()
    print(model.summary())
   # Use Nadam optimizer with variable learning rate
    optimizer = Nadam(lr=0.00001,
                  beta_1=0.9,
                  beta_2=0.999,
                  epsilon=1e-08,
                  schedule_decay=0.004)


   # Loss: MSE and Metric = RMSE
    model.compile(optimizer= optimizer, 
              loss='mean_squared_error',
              metrics=[rmse])

   #Callback to save the best model
    saveBase_Model = CNN_ModelCheckpoint(model, model_dir+"base_model_weights.h5")

   #define callback functions
    callbacks = [#EarlyStopping(monitor='val_rmse', patience=3, verbose=2),
             saveBase_Model]

    history = model.fit(x_train,
                    y_train,
                    epochs = 1000,
                    batch_size = 128,
                    validation_split = 0.15, #data = (x_test, y_test),
                    callbacks = callbacks
                    )
    
    return(model, history)

def plot_history(history):
    plt.style.use('seaborn-darkgrid')
    plt.plot(history.history['rmse'])
    plt.xlabel('Epochs')
    plt.ylabel('Root Mean Square Error')
    plt.title('Model Training Error')
    plt.show() 
    

def test_submit(model):

    lookid_data = pd.read_csv(lookid_dir)
    test_data = pd.read_csv(test_file)

    x_test = []
    for i in range(0,len(test_data)):
        img = test_data['Image'][i].split(' ')
        x_test.append(img)
    
    x_test = np.array(x_test,dtype = 'float')
    x_test = x_test/255.0
    x_test = x_test.reshape(-1,96,96,1)    

    y_test = model.predict(x_test)
    y_test = np.clip(y_test,0,96)

    lookid_list = list(lookid_data['FeatureName'])
    imageID = list(lookid_data['ImageId']-1)
    pred_list = list(y_test)

    rowid = list(lookid_data['RowId'])

    feature = []
    for f in list(lookid_data['FeatureName']):
        feature.append(lookid_list.index(f))
    
    
    submit_data = []
    for x,y in zip(imageID,feature):
        submit_data.append(pred_list[x][y])
    rowid = pd.Series(rowid,name = 'RowId')
    loc = pd.Series(submit_data,name = 'Location')
    submission = pd.concat([rowid,loc],axis = 1)
    submission.to_csv(submission_file,index = False)    
    return



def main():
    
    #start new keras session
    reset_keras() 
    
    #initialize paths/file names
    set_file_names()
    
    #Read train data and drop bad samples
    train_data = pd.read_csv(train_file) 
    train_data = train_data.drop(bad_samples).reset_index(drop=True)
    
    #get images and labels
    x_train = convert_pixels(train_data)
    y_train = train_data[[col for col in train_data.columns if col != 'Image']].to_numpy()
    
    #Fill missing values using iterative imputer
    imputer = IterativeImputer(max_iter=1000, tol=0.01, random_state=1)
    y_train = imputer.fit_transform(y_train)
    
    #Fix incorrect mouth co-ordinates
    y_train = fill_mouth_coord(y_train)
    
    #flip images and augment train set
    if add_flip_horiz:
       # Apply the augmentation and add the new data to the training set
        flipped_img,flipped_coord = flip_img_horiz(train_data)
    
        imputer = IterativeImputer(max_iter=1000, tol=0.01, random_state=3)
        flipped_coord = imputer.fit_transform(flipped_coord)
        flipped_coord = fill_mouth_coord(flipped_coord)
        x_train = np.append(x_train,flipped_img,axis=0)
        y_train = np.append(y_train,flipped_coord,axis=0)
        
    print(x_train.shape, y_train.shape)    
        
    #Rotate by +/-5, +/-10 or +/-14 degrees    
    if add_rotate_img:
        orig_x_train = x_train.copy()
        orig_y_train = y_train.copy()
    
        x_rotate, y_rotate = rotate_img(orig_x_train,orig_y_train)
        x_train = np.append(x_train,x_rotate,axis=0)
        y_train = np.append(y_train,y_rotate,axis=0)   
        
    #blur images    
    if add_blur_img:
        
        #create x and y for blurring
        x_blur = None
        y_blur = None
        x_train = np.append(x_train,blur_img(x_blur),axis=0)
        y_train = np.append(y_train,y_blur,axis=0)
     
    
    #start model training
    print("Model Training") 
    model, history = model_training(x_train, y_train)
    
    #save history as JSON file
    with open(history_dir +'history.json', 'w') as f:
        json.dump(str(history.history), f)
        
    #plot history    
    #plot_history(history)
    
    #create submission file
    test_submit(model)
    
    
    #clear session
    sess = get_session()
    clear_session()
    sess.close()
    
    try:
        del model # this is from global space - change this as you need
    except:
        print("Model clear Failed")
    print(gc.collect())  
    
    return
   
    
# call main()    
if __name__ == "__main__":
    main()
