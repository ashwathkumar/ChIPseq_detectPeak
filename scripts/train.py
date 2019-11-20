import numpy as np
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import load_model

#Directories
train_dir = 'images_from_script/train'
validation_dir = 'images_from_script/validation'

# Initialize variables used in the model.
img_width, img_height = 150, 150 #Image dimensions
train_samples = 2000 #number of training examples. 1000 in each group. 
validation_samples = 800 # number of validation examples. 400 in each group. 
epochs = 50
batch_size = 16




def save_features_from_vgg():
    vgg_model = applications.VGG16(include_top=False, weights='imagenet') #VGG16 load the model without top layer. 
    #Data generators so that I can extract the features without the top layer using weights from VGG16
    datagenerator = ImageDataGenerator(rescale=1. / 255)
    train_generator = datagenerator.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    validation_generator = datagen.flow_from_directory(
        validation_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    #Predict features
    features_train = vgg_model.predict_generator(
        train_generator, 
        train_samples // batch_size)
    features_validation = vgg_model.predict_generator(
        validation_generator, 
        validation_samples // batch_size)
    #since we have equal number of samples in both categories and we used shuffle = FALSE in our datagenerator above. Thus, we can just make the labels in this case.  
    train_labels = np.array([0] * int(train_samples / 2) + [1] * int(train_samples / 2))
    validation_labels = np.array([0] * int(validation_samples / 2) + [1] * int(validation_samples / 2))
    return features_train,train_labels,features_validation,validation_labels


def train_last_layer(features_train,train_labels,features_validation,validation_labels):
    #Make the top layer. 
    model = Sequential()
    model.add(Flatten(input_shape=features_train.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))#signmoid since we have only two classes in our case - Peak or no peak. 

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', 
                  metrics=['accuracy']) #compile with rmsprop. We can also use Adam - seems to be doing about the same performance. 
    #Train the top layer with training and validation data. 
    model.fit(features_train, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(features_validation, validation_labels))
    model.save_weights('vgg16_transfer_top_layer.h5')
    model.save('vgg16_transfer_top_model.h5')
    
if __name__ == '__main__':
    features_train,train_labels,features_validation,validation_labels = save_features_from_vgg()
    train_last_layer(features_train,train_labels,features_validation,validation_labels)
    
