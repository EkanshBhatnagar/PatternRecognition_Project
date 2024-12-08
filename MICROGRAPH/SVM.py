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

def result_record(file, train_acc, valid_acc, test_acc, svclassifier):
    file.write('Training: \n')
    file.write('accuracy: {:.3f}% \n'.format(train_acc * 100))
    file.write('Validation: \n')
    file.write('accuracy: {:.3f}% \n'.format(valid_acc * 100))
    file.write('Test: \n')
    file.write('accuracy: {:.3f}% \n'.format(test_acc * 100))
    # file.write('The total number of support vectors: \n')
    # file.write(str(np.sum(svclassifier.n_support_)) + '\n')
    file.write('\n')


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
        img = load_img(full_path, target_size=(224, 224))  # Resize for simplicity
        img_array = img_to_array(img)
        img_array = img_array.flatten() / 255.0  # Flatten and normalize
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

time1 = time.time()
model = SVM(degree=8)
svclassifier, train_acc = model.train(X, y_encoded)
#eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
test_acc = model.evaluate(X_test, y_test_encoded, svclassifier)
time2 = time.time()
print('SVM with polynomial kernel (degree=8) \n')
with open('result record.txt', 'a') as f:
    f.write('SVM with polynomial kernel (degree=8, original MNIST dataset) \n')
    f.write('time to run this part: {:.3f}s \n'.format(time2 - time1))
    result_record(f, train_acc, 0, test_acc, svclassifier)