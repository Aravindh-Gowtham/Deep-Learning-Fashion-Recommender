import keras
import tensorflow
import numpy as np
import pickle
from numpy.linalg import norm
import cv2
from keras import utils
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet import ResNet50, preprocess_input
from sklearn.neighbors import NearestNeighbors

img_feats_lst = pickle.load(open('image_features.pkl', 'rb'))
imagefiles = pickle.load(open('image_files.pkl', 'rb'))

fashion_mdl = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
fashion_mdl.trainable = False

fashion_mdl = keras.Sequential([
    fashion_mdl, GlobalMaxPooling2D()
])

temp_image = utils.load_img('test-images/mens-tee.jpg', target_size=(224, 224))
temp_image_array = utils.img_to_array(temp_image)
mod_array = np.expand_dims(temp_image_array, axis=0)
img_preprocessed = preprocess_input(mod_array)
image_features = fashion_mdl.predict(img_preprocessed).flatten()
norm_image_feats = image_features/norm(image_features)

recos_mdl = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
recos_mdl.fit(img_feats_lst)

dist, idxs = recos_mdl.kneighbors([norm_image_feats])


for file in idxs[0][1:6]:
    temp_img = cv2.imread(imagefiles[file])
    cv2.imshow('output', cv2.resize(temp_img, (512, 512)))
    cv2.waitKey(0)
