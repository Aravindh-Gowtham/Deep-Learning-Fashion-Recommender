import numpy as np
import os
import pickle
from numpy.linalg import norm
from tqdm import tqdm
import tensorflow
from keras import utils
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50, preprocess_input

fashion_pred_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
fashion_pred_model.trainable = False

fashion_pred_model = tensorflow.keras.Sequential([
    fashion_pred_model, GlobalMaxPooling2D()
])


def get_image_feats(img_file, inp_model):
    temp_image = utils.load_img(img_file, target_size=(224, 224))
    temp_image_array = utils.img_to_array(temp_image)
    mod_array = np.expand_dims(temp_image_array, axis=0)
    img_preprocessed = preprocess_input(mod_array)
    image_features = inp_model.predict(img_preprocessed).flatten()
    norm_image_feats = image_features/norm(image_features)
    return norm_image_feats

#Below lines are commented after dumping the info into pickle files

#imagefiles = []
#for f in os.listdir('images'):
#    imagefiles.append(os.path.join('images', f))

#image_feats_lst = []
#for imgfile in tqdm(imagefiles):
#    image_feats_lst.append(get_image_feats(imgfile, fashion_pred_model))

#pickle.dump(image_feats_lst, open('image_features.pkl', 'wb'))
#pickle.dump(imagefiles, open('image_files.pkl', 'wb'))
