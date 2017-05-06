import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout, Activation
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle



EPOCH = 15
batch_size = 32

def generator(data_set, batch_size):
    data_set_size = len(data_set)
    while 1:
    	# Shuffle data_set to avoid overfitting
        shuffle(data_set)
        # For every batch of 32 data_set
        for batch_index in range(0, data_set_size, batch_size):
        	# Capture all 32 data_set in this batch inside a variable
            batch_data_32 = data_set[batch_index:batch_index+batch_size]
            # Create lists to hold images and angles
            imgs = []
            angles = []
            # For every sample in the current batch (row in the csv file)
            for batch_row in batch_data_32:
            	# For sample 0, 1, and 2 (row 1, 2, and 3 in csv)
                for cell_index in range(3):
                	# Save the file path from the csv file into a variable
                    path_from_root = batch_row[cell_index]
                    # Get the file name from the file path
                    file_name = path_from_root.split('/')[-1]
                    # Save the file path from model to image
                    model_to_img_path = 'data/IMG/' + file_name
                    # Save the image in a variable
                    img = cv2.imread(model_to_img_path)
                    # Add the image to the list of images
                    imgs.append(img)
                    # Get the data from the row's 'angle' column
                    angle = float(batch_row[3])
                    # If it is a left image
                    if cell_index == 1:
                    	# Steer right
                        angle += 0.1
                    # If it is a right image
                    elif cell_index == 2:  
                    	# Steer left
                        angle -= 0.1
                    # Add the new angle to the list
                    angles.append(angle)
            # Save the images into a numpy array
            x_train = np.array(imgs)
            # Save the angles into a numpy array
            y_train = np.array(angles)
            # Save the generator
            yield shuffle(x_train, y_train)



# Get rows from driving log csv file
rows = []
with open('data/driving_log.csv') as csvfile:
	# Read csv file into program
    data_file = csv.reader(csvfile)
    # For each row of the csv file, add it to the rows array
    for row in data_file:
        rows.append(row)

# Split the image dataset for validation
training_set, validation_set = train_test_split(rows, test_size=0.2)

# Prepare generators for training and validation sets
train_generator = generator(training_set, batch_size)
validation_generator = generator(validation_set, batch_size)



# Nvidia neural network architecture
model = Sequential()

# Layer Lambda - Normalization & Cropping
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0))))

# Layer 1 - Convolutional
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(1))

# Layer 2 - Convolutional
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(1))

# Layer 3 - Convolutional
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Dropout(1))

# Layer 4 - Convolutional
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(1))

# Layer 5 - Convolutional
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(1))

# Layer 6 - Flatten
model.add(Flatten())

# Layer 7 - Fully Connected
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 8 - Fully Connected
model.add(Dense(50))
model.add(Activation('relu'))

# Layer 9 - Fully Connected
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 10 - Fully Connected
model.add(Dense(1))



# train the model
model.compile(loss='mse', optimizer='adam')
fit_generator = model.fit_generator(
    train_generator, samples_per_epoch=len(training_set)*3,
    validation_data=validation_generator, nb_val_samples=len(validation_set)*3,
    nb_epoch=EPOCH)


# save model
model.save('model.h5')
