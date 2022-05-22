import numpy as np
import tensorflow
import pickle
import streamlit as st
from PIL import Image
import os
import keras
from numpy.linalg import norm
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

st.title('Deep Learning Fashion Recommender')


def storeimagefile(inp_file):
    try:
        with open(os.path.join('input-images', inp_file.name), 'wb') as f:
            f.write(inp_file.getbuffer())
        return 1
    except:
        return 0


def get_image_feats(image_loc, model):
    temp_image = utils.load_img(image_loc, target_size=(224, 224))
    temp_array = utils.img_to_array(temp_image)
    mod_array = np.expand_dims(temp_array, axis=0)
    final_image = preprocess_input(mod_array)
    image_feat = model.predict(final_image).flatten()
    norm_image_feats = image_feat / norm(image_feat)

    return norm_image_feats


def deep_recommender(image_feats, image_feats_list):
    sim_images = NearestNeighbors(n_neighbors=6, algorithm='brute', metric='euclidean')
    sim_images.fit(image_feats_list)
    dist, idxs = sim_images.kneighbors([image_feats])
    return idxs


input_file = st.file_uploader("Upload an image (related to fashion!!) and get similar product recommendations")
if input_file is not None:
    if storeimagefile(input_file):
        show_im = Image.open(input_file)
        st.image(show_im)
        im_feats = get_image_feats(os.path.join("input-images", input_file.name), fashion_mdl)
        final_idxs = deep_recommender(im_feats, img_feats_lst)
        res1, res2, res3, res4, res5 = st.columns(5)

        with res1:
            st.image(imagefiles[final_idxs[0][1]])
        with res2:
            st.image(imagefiles[final_idxs[0][2]])
        with res3:
            st.image(imagefiles[final_idxs[0][3]])
        with res4:
            st.image(imagefiles[final_idxs[0][4]])
        with res5:
            st.image(imagefiles[final_idxs[0][5]])

    else:
        st.header("Trouble uploading the chosen file! Please re-upload")
