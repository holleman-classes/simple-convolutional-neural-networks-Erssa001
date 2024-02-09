### Add lines to import modules as needed
#%%
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Dropout, Activation, MaxPooling2D, Flatten, Dense, Add
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions
print(tf.__version__)
##
#%%

def build_model1():
  model = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3), strides=(2, 2),activation="relu", padding='same'),
    BatchNormalization(),

    Conv2D(128, kernel_size=(3, 3),activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3),activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3),activation="relu", padding='same'),
    BatchNormalization(),
    Conv2D(128, kernel_size=(3, 3),activation="relu", padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),

    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),

    Dense(10)
  ]) # Add code to define model 1.
  return model

def build_model2():
  model = tf.keras.Sequential([
    Input(shape=(32, 32, 3)),
    Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same'),
    BatchNormalization(),

    DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same', use_bias=False),
    BatchNormalization(),
    DepthwiseConv2D(kernel_size=(3, 3), strides=(2, 2), activation="relu", padding='same', use_bias=False),
    BatchNormalization(),

    DepthwiseConv2D(kernel_size=(3, 3), activation="relu", padding='same', use_bias=False),
    BatchNormalization(),
    DepthwiseConv2D(kernel_size=(3, 3), activation="relu", padding='same', use_bias=False),
    BatchNormalization(),
    DepthwiseConv2D(kernel_size=(3, 3), activation="relu", padding='same', use_bias=False),
    BatchNormalization(),
    DepthwiseConv2D(kernel_size=(3, 3), activation="relu", padding='same', use_bias=False),
    BatchNormalization(),
    MaxPooling2D(pool_size=(4, 4), strides=(4, 4)),
    
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    
    Dense(10)
  ]) # Add code to define model 2.
  return model

def build_model3():
  # Input layer
  input_layer = Input(shape=(32, 32, 3))

  # Block 1
  x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(input_layer)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
  y = x

  # Block 2
  x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  # Block 3
  x = Conv2D(128, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  # Res Block 1
  y = Conv2D(128, kernel_size=(1,1), strides=(4, 4), name='SkipConvA')(y)
  y = Add()((x,y))

  # Block 4
  x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(y)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  # Block 5
  x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  # Res Block 2
  y = Add()((x,y))

  # Block 6
  x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(y)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  # Block 7
  x = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  # Res Block 3
  y = Add()((x,y))
  
  x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)

  # Dense layers
  x = Flatten()(x)
  x = Dense(128, activation='relu')(x)
  x = BatchNormalization()(x)

  # Output layer
  output_layer = Dense(10)(x)

  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
  ## This one should use the functional API so you can create the residual connections
  return model

def build_model50k():
  # Input layer
  input_layer = Input(shape=(32, 32, 3))

  # Block 1
  x = Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(input_layer)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
  y = x

  # Block 2
  x = Conv2D(48, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(x)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)

  # Res Block 1
  y = Conv2D(48, kernel_size=(1,1), strides=(2,2), name='SkipConvA')(y)
  y = Add()((x,y))

  # Block 3
  x = Conv2D(64, kernel_size=(3, 3), strides=(2, 2), activation='relu', padding='same')(y)
  x = BatchNormalization()(x)
  x = Dropout(0.2)(x)
    
  x = MaxPooling2D(pool_size=(4, 4), strides=(4, 4))(x)

  # Dense layers
  x = Flatten()(x)
  x = Dense(64, activation='relu')(x)
  x = BatchNormalization()(x)

  # Output layer
  output_layer = Dense(10)(x)

  model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
   # Add code to define model 4.
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

    train_labels = train_labels.squeeze()
    test_labels = test_labels.squeeze()
    val_labels = val_labels.squeeze()

    train_images = train_images / 255.0
    test_images  = test_images  / 255.0
    val_images   = val_images   / 255.0
    ########################################
    ## Build and train model 1
    model1 = build_model1()
    
    # compile and train model 1.
    model1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model1.summary()
    # train_hist = model1.fit(train_images, train_labels, 
    #                         validation_data=(val_images, val_labels), # or use `validation_split=0.1`
    #                         epochs=(50))
    # model1.save('model1.h5')

    # Testing Model1
    # image_path = 'test_image_dog.png'
    # img = image.load_img(image_path, target_size=(32, 32))
    # img_array = image.img_to_array(img)
    # img_array = np.expand_dims(img_array, axis=0)
    # img_array = preprocess_input(img_array)

    # # Make predictions
    # predictions = model1.predict(img_array)

    # # Get the predicted class index
    # top3_classes = np.argsort(predictions[0])[::-1][:3]
    # for i, class_index in enumerate(top3_classes):
    #   class_name = class_names[class_index]
    #   probability = predictions[0, class_index]
    #   print(f"Top {i + 1}: {class_name} ({probability:.2f})")

    ## Build, compile, and train model 2 (DS Convolutions)
    model2 = build_model2()
    model2.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model2.summary()
    # train_hist = model2.fit(train_images, train_labels, 
    #                         validation_data=(val_images, val_labels), # or use `validation_split=0.1`
    #                         epochs=50)
    # model2.save('model2.h5')
    ## Repeat for model 3 and your best sub-50k params model
  
    model3 = build_model3()
    
    # compile and train model 1.
    model3.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model3.summary()
    # train_hist = model3.fit(train_images, train_labels, 
    #                         validation_data=(val_images, val_labels), # or use `validation_split=0.1`
    #                         epochs=(50))
    # model3.save('model3.h5')

    model50k = build_model50k()
    
    # compile and train model 1.
    model50k.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    model50k.summary()
    # train_hist = model50k.fit(train_images, train_labels, 
    #                         validation_data=(val_images, val_labels), # or use `validation_split=0.1`
    #                         epochs=(50))
    # model50k.save('best_model.h5')

# %%
