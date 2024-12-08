import numpy as np
import time
from Model import SVM, TISVM, RISVM, LOCSVM, LTISVM, LRISVM, TIRISVM, LTIRISVM, RIISVM, KNN, TDSVM, ResNetModel
from DataLoader import load_data, train_valid_split
import ctypes
import torch
import pytorch_lightning as pl
import torch.utils.data as data_utils
from tqdm.autonotebook import tqdm
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.models import Model

# Load the CSV file
file_path = 'micrograph.csv'
df = pd.read_csv(file_path)

# Select the first 100 entries
subset_df = df[['path', 'primary_microconstituent']].head(100)

# Get the lists of image paths and labels from the sample
sample_image_paths = subset_df['path'].tolist()
sample_labels = subset_df['primary_microconstituent'].tolist()

test_set_df = df[['path', 'primary_microconstituent']].iloc[100:600]
test_image_paths = test_set_df['path'].tolist()
test_labels = test_set_df['primary_microconstituent'].tolist()

# Define image directory
image_dir = 'images/'

# Preprocess images
def preprocess_images(image_paths):
    images = []
    for img_path in image_paths:
        full_path = os.path.join(image_dir, img_path)
        img = load_img(full_path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)

X = preprocess_images(sample_image_paths)
X_test = preprocess_images(test_image_paths)


# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(df['primary_microconstituent'])
y_encoded = label_encoder.transform(sample_labels)
y_test_encoded = label_encoder.transform(test_labels)

# One-hot encode labels
y_categorical = to_categorical(y_encoded)
y_test_categorical = to_categorical(y_test_encoded)
input_tensor = Input(shape=(224, 224, 3))

base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)

x = base_model.output
x = GlobalAveragePooling2D()(x)

# Adjust the number of classes according to your dataset
num_classes = len(label_encoder.classes_)

predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X, y_categorical, batch_size=16, epochs=10, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test_categorical)

print(f'Test Accuracy: {accuracy * 100:.2f}%')