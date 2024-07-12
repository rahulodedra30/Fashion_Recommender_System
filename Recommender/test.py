import pickle
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore
from app import build_model, extract_features

features_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('file_names.pkl','rb'))

"""Build the ResNet50 model without the top layer and add a GlobalMaxPooling layer."""
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

result1 = extract_features('test_images/fashionable-women-saree-with-latest-design.jpg',model)

neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(features_list)

distances, indices = neighbors.kneighbors([result1])
print(indices)

for files in indices[0]:
    temp_img = cv2.imread(filenames[files])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)