import numpy as np
from keras.preprocessing.image import ImageDataGenerator,array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.models import load_model


def predict_image(image,model_file):
    img = load_img(image,target_size=(150,150))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    vgg_model = applications.VGG16(include_top=False, weights='imagenet')
    model = load_model(model_file)
    features = vgg_model.predict(img)
    predictVal = model.predict(features)
    if predictVal[0][0] == 0:
        print ("It is not a peak.")
    else:
        print ("It is a peak.")
		

if __name__ == '__main__':
    predict_image(sys.argv[1],sys.argv[2])