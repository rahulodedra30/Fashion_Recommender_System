import os
import streamlit as st
from PIL import Image
import pickle
import numpy as np
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore

features_list = np.array(pickle.load(open('embeddings.pkl','rb')))
filenames = pickle.load(open('file_names.pkl','rb'))

# Build the ResNet50 model without the top layer and add a GlobalMaxPooling layer.
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
model = tf.keras.Sequential([
    base_model,
    GlobalMaxPooling2D()
])

def extract_features(img_path, model):
    """Extract features from an image using the given model."""
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / np.linalg.norm(result)
    return normalized_result

def recommend(features,features_list):
    neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
    neighbors.fit(features_list)
    distances, indices = neighbors.kneighbors([features])
    return indices

st.title('Fashion Recommender System')

uploads_dir = 'uploads'
if not os.path.exists(uploads_dir):
    os.makedirs(uploads_dir)

def save_uploaded_file(uploaded_file):
    try:
        with open(os.path.join('uploads', uploaded_file.name), 'wb') as f:
            f.write(uploaded_file.getbuffer())
            return True
    except Exception as e:
        st.error(f"Error: {e}")
        return False

uploaded_file = st.file_uploader("Choose an image")
if uploaded_file is not None:
    if save_uploaded_file(uploaded_file):
        # st.success("File uploaded successfully")
        display_img = Image.open(uploaded_file)
        # display_img = display_img.resize((400, 300))  # Resize to desired dimensions
        st.image(display_img)
        features = extract_features(os.path.join('uploads',uploaded_file.name),model)
        indices = recommend(features,features_list)
        col1,col2,col3,col4,col5 = st.columns(5)

        with col1:
            recommended_img1 = Image.open(filenames[indices[0][0]])
            resized_img1 = recommended_img1.resize((800, 800)) 
            st.image(resized_img1)

        with col2:
            recommended_img2 = Image.open(filenames[indices[0][1]])
            resized_img2 = recommended_img2.resize((400, 400))  
            st.image(resized_img2)

        with col3:
            recommended_img3 = Image.open(filenames[indices[0][2]])
            resized_img3 = recommended_img3.resize((400, 400))  
            st.image(resized_img3)

        with col4:
            recommended_img4 = Image.open(filenames[indices[0][3]])
            resized_img4 = recommended_img4.resize((400, 400))  
            st.image(resized_img4)

        with col5:
            recommended_img5 = Image.open(filenames[indices[0][4]])
            resized_img5 = recommended_img5.resize((400, 400))  
            st.image(resized_img5)
    else:   
        st.error("Error occurred in uploading file")