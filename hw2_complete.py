### Add lines to import modules as needed
#%%
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras import Input, layers
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
print(tf.__version__)
##
#%%

def build_model1():
  model = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3), strides=(2, 2),activation="relu", padding='same'),
    layers.BatchNormalization(),

    layers.Conv2D(128, kernel_size=(3, 3),activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3),activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3),activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.Conv2D(128, kernel_size=(3, 3),activation="relu", padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),

    layers.Dense(10)
  ]) # Add code to define model 1.
  return model

def build_model2():
  model = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    layers.Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same'),
    layers.BatchNormalization(),

    layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same', use_bias=False),
    layers.BatchNormalization(),

    layers.DepthwiseConv2D(kernel_size=(3, 3), activation="relu", padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.DepthwiseConv2D(kernel_size=(3, 3), activation="relu", padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.DepthwiseConv2D(kernel_size=(3, 3), activation="relu", padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.DepthwiseConv2D(kernel_size=(3, 3), activation="relu", padding='same', use_bias=False),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    
    layers.Dense(10)
  ]) # Add code to define model 2.
  return model

def build_model3():
  model = None # Add code to define model 1.
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  model = None # Add code to define model 1.
  return model
#%%
# no training or dataset construction should happen above this line
if __name__ == '__main__':

    ########################################
    ## Add code here to Load the CIFAR10 data set
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    class_names = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

    # Now separate out a validation set.
    val_frac = 0.1
    num_val_samples = int(len(train_images)*val_frac)
    # choose num_val_samples indices up to the size of train_images, !replace => no repeats
    val_idxs = np.random.choice(np.arange(len(train_images)), size=num_val_samples, replace=False)
    trn_idxs = np.setdiff1d(np.arange(len(train_images)), val_idxs)
    val_images = train_images[val_idxs, :,:,:]
    train_images = train_images[trn_idxs, :,:,:]

    val_labels = train_labels[val_idxs]
    train_labels = train_labels[trn_idxs]
    ########################################
    #%%
    ## Build and train model 1
    model1 = build_model1()
    
    # compile and train model 1.
    model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model1.summary()
    train_hist = model1.fit(train_images, train_labels, 
                            validation_data=(val_images, val_labels), # or use `validation_split=0.1`
                            epochs=(10))
    model1.save('model1.h5')
    #%%
    #Testing Model1
    image_path = 'test_image_dog.png'
    img = image.load_img(image_path, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Make predictions
    predictions = model1.predict(img_array)

    # Get the predicted class index
    top3_classes = np.argsort(predictions[0])[::-1][:3]
    for i, class_index in enumerate(top3_classes):
      class_name = class_names[class_index]
      probability = predictions[0, class_index]
      print(f"Top {i + 1}: {class_name} ({probability:.2f})")
    #%%
    ## Build, compile, and train model 2 (DS Convolutions)
    model2 = build_model2()
    model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model2.summary()
    train_hist = model2.fit(train_images, train_labels, 
                            validation_data=(val_images, val_labels), # or use `validation_split=0.1`
                            epochs=7)
    model2.save('model2.h5')
   #%%
    #Testing Model2
    # Make predictions
    predictions = model2.predict(img_array)

    # Get the predicted class index
    top3_classes = np.argsort(predictions[0])[::-1][:3]
    for i, class_index in enumerate(top3_classes):
      class_name = class_names[class_index]
      probability = predictions[0, class_index]
      print(f"Top {i + 1}: {class_name} ({probability:.2f})")
    ### Repeat for model 3 and your best sub-50k params model
# %%
