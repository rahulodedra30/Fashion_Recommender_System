import os
import ssl
import certifi
import pickle
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.preprocessing import image # type: ignore
from tensorflow.keras.layers import GlobalMaxPooling2D # type: ignore
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input # type: ignore

def configure_ssl():
    """Configure SSL context to use certifi's certificate bundle."""
    ssl._create_default_https_context = ssl._create_unverified_context

def build_model():
    """Build the ResNet50 model without the top layer and add a GlobalMaxPooling layer."""
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

def load_image_filenames(directory):
    """Load image filenames from a given directory."""
    filenames = [os.path.join(directory, file) for file in os.listdir(directory)]
    return filenames

def save_embeddings(filenames, feature_list, filenames_path='file_names.pkl', embeddings_path='embeddings.pkl'):
    """Save filenames and feature embeddings to pickle files."""
    with open(filenames_path, 'wb') as f:
        pickle.dump(filenames, f)
    with open(embeddings_path, 'wb') as f:
        pickle.dump(feature_list, f)

def main():
    configure_ssl()
    model = build_model()

    image_directory = 'images'
    filenames = load_image_filenames(image_directory)

    feature_list = []
    for file in tqdm(filenames):
        feature_list.append(extract_features(file, model))

    save_embeddings(filenames, feature_list)

if __name__ == "__main__":
    main()
