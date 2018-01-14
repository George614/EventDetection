# -*- coding: utf-8 -*-
import pickle
import numpy as np
from keras.layers import Input, Dense, Flatten, Dropout, Bidirectional, TimeDistributed
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.optimizers import Adam, RMSprop
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.models import Model
import cv2


PATH_TO_DATASET = 'D:\Dataset\Event detection'
DATASET = pickle.load(open(PATH_TO_DATASET + 'Dataset.p', "rb"))

'''
imageSeq = []
HeadVelSeq = []
HeadAngSeq = []
LabelSeq = []
for participant in DATASET:
    for sequence in participant:
        PathToImages =
        ImageNum = 
        for i in len(labels):
            image = cv2.imread(PathToImages + ImageNum[i])
            imageSeq.append(image)
        HeadVelSeq.append(HeadVelocity)
        HeadAngSeq.append(HeadAngVelo)
        LabelSeq.append(labels)
'''
'''
Read images
'''
def readImage(data):
    PathToImages = data[:,-2]
    ImageNum = data[:,-1]
    images = []
    for i in range(len(ImageNum)):
        image = cv2.imread(PathToImages[i] + ImageNum[i])
        image = image[:,:,0]
        images.append(image)
    return images

def Generate_Dataset(DATASET, Participant_Index, WinSize):
    '''Given the Type_str as 'Train' or 'Test', this function will generate
    a dataset from the given participant index. For multiple participants,
    provide a list with their index numbers.'''

    Forward_Batches = []
    Labels = []
    Backward_Batches = []
    image_F = []
    image_B = []
    for i in Participant_Index:
        for j in range(0, len(DATASET[i])):
            temp_batch_forward, temp_batch_backward, temp_label = Get_Batches(DATASET[i][j], WinSize)
            image = temp_batch_forward
            images_F = readImage(temp_batch_forward)
            images_B = readImage(temp_batch_backward)
            Forward_Batches = Forward_Batches + temp_batch_forward
            Labels = Labels + temp_label
            Backward_Batches = Backward_Batches + temp_batch_backward
    return (Forward_Batches, Backward_Batches, Labels)

def Get_Batches(ip, WinSize):
    """Given input sequence, convert it to batches and return the data"""
    data = np.concatenate((ip["HeadVelocity"]，ip['HAng_AzEl_vel'],ip['PathToRightEyeImages'],ip['EyeFrameNo']), axis=1)
    labels = ip["Labels"]
    
    num_samples = np.shape(data)[0]
    forward_batch = []
    backward_batch = []
    label_batch = []
    for i in range(WinSize, num_samples - WinSize + 2):
        x = i - WinSize
        y = i + WinSize - 1
        forward_batch.append(data[x:i, :])
        label_batch.append(labels[i-1, :])
        backward_batch.append(np.flip(data[i-1:y, :], axis=0))
    return (forward_batch, backward_batch, label_batch)

def Divide_Dataset(x, test_per):
    """Given labels, divide the dataset by the given fraction and ensure
    that each class has representation in the testing set"""
    Label_types = np.unique(x)
    train_idx = []
    test_idx = []
    perClass = []
    for i in Label_types:
        idx_loc = np.where(x == i)[0]
        L = np.size(idx_loc)
        perClass.append(L)
        temp = np.random.permutation(L)
        test_locs = temp[0:int(np.round(test_per*L))]
        train_locs = temp[int(np.round(test_per*L)):]
        test_idx.append(idx_loc[test_locs])
        train_idx.append(idx_loc[train_locs])
    return (np.concatenate(train_idx).astype('int'), np.concatenate(test_idx).astype('int'), perClass)





FORWARD, BACKWARD, LABELS = Generate_Dataset(DATASET, [0, 1, 2], 5)
LABELS = np.asarray(LABELS).squeeze()
TRAIN_IDX, TEST_IDX = Divide_Dataset(LABELS, 0.3)

FORWARD_TRAIN = [FORWARD[i] for i in TRAIN_IDX]
BACKWARD_TRAIN = [BACKWARD[i] for i in TRAIN_IDX]
LABELS_TRAIN = [LABELS[i] for i in TRAIN_IDX]

FORWARD_TEST = [FORWARD[i] for i in TEST_IDX]
BACKWARD_TEST = [BACKWARD[i] for i in TEST_IDX]
LABELS_TEST = [LABELS[i] for i in TEST_IDX]

'''
Keras Code for the architecture
'''

head_vel = Input(shape =(5,))
head_ang_vel = Input(shape =(5,))

'''
x = Conv2D(16, (3, 3),padding='same', activation='relu')(input1)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Dropout(0.25)(x)

x = Conv2D(32, (3, 3),padding='same', activation='relu')(x)
x = Conv2D(32, (3, 3),padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)

x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
x = MaxPooling2D(pool_size=(2,2))(x)
x = Dropout(0.25)(x)
'''

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model = Sequential()
vision_model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=(320, 240, 1)))
vision_model.add(Conv2D(16, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(32, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
vision_model.add(Conv2D(64, (3, 3), activation='relu'))
vision_model.add(Conv2D(128, (3, 3), activation='relu'))
vision_model.add(MaxPooling2D((2, 2)))
vision_model.add(Flatten())

# Now let's get a tensor with the input of our vision model:
image_input_F = Input(shape = (5,320, 240, 1))
image_input_B = Input(shape = (5,320, 240, 1))
video_input = Input(shape = (-1,320,240,1))
frame_sequence_F = []
frame_sequence_B = []
for i in range(5):    
    encoded_image_F = vision_model(image_input_F[i])
    encoded_image_B = vision_model(image_input_B[i])
    frame_sequence_F.append(encoded_image_F)
    frame_sequence_B.append(encoded_image_B)

'''
encoded_frame_sequence_F = TimeDistributed(vision_model)(image_input_F)
encoded_frame_sequence_B = TimeDistributed(vision_model)(image_input_B) 
'''
encoded_frame_sequence_B = biLSTM(frame_sequence_B)
encoded_frame_sequence_F = biLSTM(frame_sequence_F)
encoded_headvel = biLSTM(head_vel)
encoded_headangvel = biLSTM(head_ang_vel)

# merge image data and head velocity data together
merged = keras.layers.concatenate([encoded_frame_sequence_F,encoded_frame_sequence_B,encoded_headvel,encoded_headangvel])
pre_output = Dense(100,activation='relu')(merged)
pre_output = Dropout(0.5)(pre_output)
output = Dense(3, activation='softmax')(pre_output)

# LSTM model 
biLSTM = Sequential()
biLSTM.add(LSTM(32, return_sequences=True, input_shape =(time_sep,256)))
biLSTM.add(LSTM(32, return_sequences=True))
biLSTM.add(LSTM(32))

gaze_classifier = Model(inputs=[image_input_F,image_input_B,head_vel,head_ang_vel],outputs=output)

gaze_classifier.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

gaze_classifier.train_on_batch([FORWARD_TRAIN,BACKWARD_TRAIN],LABELS_TEST,sample_weight=None)
#gaze_classifier.fit([FORWARD_TRAIN,BACKWARD_TRAIN],LABELS_TEST,epochs=500)

loss_and_metrics = model.evaluate(FORWARD_TEST, LABELS_TEST, batch_size=64)
