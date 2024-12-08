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

# Preprocess images
def preprocess_images(image_paths):
    images = []
    for img_path in image_paths:
        full_path = os.path.join(image_dir, img_path)
        img = load_img(full_path, target_size=(224, 224))
        img_array = img_to_array(img)
        images.append(img_array)
    return np.array(images)
    
def preprocess_images_svm(image_paths):
    images = []
    for img_path in image_paths:
        full_path = os.path.join(image_dir, img_path)
        img = load_img(full_path, target_size=(28, 28),color_mode='grayscale')  # Resize for simplicity
        img_array = img_to_array(img)
        img_array = img_array.flatten() / 255.0  # Flatten and normalize
        images.append(img_array)
    return np.array(images)

# Load the CSV file
file_path = 'micrograph.csv'
df = pd.read_csv(file_path)

samples_num = [100, 200, 400]
results = open("result.txt","w",buffering=1)
results.write("model,number_of_samples,test_accuracy\n")

for num in samples_num:
    # Select the first 100 entries
    subset_df = df[['path', 'primary_microconstituent']].head(num)

    # Get the lists of image paths and labels from the sample
    sample_image_paths = subset_df['path'].tolist()
    sample_labels = subset_df['primary_microconstituent'].tolist()

    test_set_df = df[['path', 'primary_microconstituent']].iloc[num:num+500]
    test_image_paths = test_set_df['path'].tolist()
    test_labels = test_set_df['primary_microconstituent'].tolist()

    # Define image directory
    image_dir = 'images/'

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

    #Adjust the number of classes according to your dataset
    num_classes = len(label_encoder.classes_)
    predictions = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X, y_categorical, batch_size=16, epochs=10, validation_split=0.2)
    loss, accuracy = model.evaluate(X_test, y_test_categorical)

    print("RESNET50:")
    print(f'Test Accuracy with {num} samples: {accuracy * 100:.2f}%')
    results.write("RESNET50,"+str(num)+","+str(accuracy * 100)+"\n")

    X = preprocess_images_svm(sample_image_paths)
    X_test = preprocess_images_svm(test_image_paths)
    # Encode labels
    label_encoder = LabelEncoder()
    label_encoder.fit(df['primary_microconstituent'])
    y_encoded = label_encoder.transform(sample_labels)
    y_test_encoded = label_encoder.transform(test_labels)
    
    # One-hot encode labels
    y_categorical = to_categorical(y_encoded)
    y_test_categorical = to_categorical(y_test_encoded)
    
    model_svm = SVM(degree=8)
    model_tisvm = TISVM(degree=8)
    model_locsvm = LOCSVM(degree=9, filter=9)
    model_lrisvm = LRISVM(degree=8, filter=3)
    model_ltisvm = LTISVM(degree=8,filter=7)
    model_risvm = RISVM(degree=8)
    model_riisvm = RIISVM(degree=8)
    model_tirisvm = TIRISVM(degree=8)
    model_ltirisvm = LTIRISVM(degree=8,filter=7)

    time1 = time.time()
    svclassifier, train_acc = model_svm.train(X, y_encoded)
    #eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
    test_acc = model_svm.evaluate(X_test, y_test_encoded, svclassifier)
    time2 = time.time()
    print("SVM (",time2-time2,"}s):")
    print(f'Test Accuracy with {num} samples: {test_acc * 100:.2f}%')
    results.write("SVM,"+str(num)+","+str(test_acc * 100)+"\n")

    time1 = time.time()
    svclassifier, train_acc = model_tisvm.train(X, y_encoded)
    #eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
    test_acc = model_tisvm.evaluate(X_test, y_test_encoded, svclassifier)
    time2 = time.time()
    print('TISVM (',time2-time2,'}s):')
    print(f'Test Accuracy with {num} samples: {test_acc * 100:.2f}%')
    results.write("TISVM,"+str(num)+","+str(test_acc * 100)+"\n")

    time1 = time.time()
    svclassifier, train_acc = model_locsvm.train(X, y_encoded)
    #eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
    test_acc = model_locsvm.evaluate(X_test, y_test_encoded, svclassifier)
    time2 = time.time()
    print('LOCSVM ((',time2-time2,'}s):')
    print(f'Test Accuracy with {num} samples: {test_acc * 100:.2f}%')
    results.write("LOCSVM,"+str(num)+","+str(test_acc * 100)+"\n")

    time1 = time.time()
    svclassifier, train_acc = model_lrisvm.train(X, y_encoded)
    #eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
    test_acc = model_lrisvm.evaluate(X_test, y_test_encoded, svclassifier)
    time2 = time.time()
    print('LRISVM ((',time2-time2,'}s):')
    print(f'Test Accuracy with {num} samples: {test_acc * 100:.2f}%')
    results.write("LRISVM,"+str(num)+","+str(test_acc * 100)+"\n")

    time1 = time.time()
    svclassifier, train_acc = model_ltisvm.train(X, y_encoded)
    #eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
    test_acc = model_ltisvm.evaluate(X_test, y_test_encoded, svclassifier)
    time2 = time.time()
    print('LTISVM ((',time2-time2,'}s):')
    print(f'Test Accuracy with {num} samples: {test_acc * 100:.2f}%')
    results.write("LTISVM,"+str(num)+","+str(test_acc * 100)+"\n")

    #model_ltirisvm

    time1 = time.time()
    svclassifier, train_acc = model_ltirisvm.train(X, y_encoded)
    #eval_acc = model.evaluate(x_valid[0:val_list[idx], :], y_valid[0:val_list[idx]], svclassifier)
    test_acc = model_ltirisvm.evaluate(X_test, y_test_encoded, svclassifier)
    time2 = time.time()
    print('LTIRISVM ((',time2-time2,'}s):')
    print(f'Test Accuracy with {num} samples: {test_acc * 100:.2f}%')
    results.write("LTIRSVM,"+str(num)+","+str(test_acc * 100)+"\n")