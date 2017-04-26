import csv
import cv2
import numpy as np 

lines = []
with open('data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = 'data/IMG/' + filename
	image = cv2.imread(current_path)
	images.append(image)
	measurement = float(line[3]) # Steering measurement
	measurements.append(measurement)

x_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Lambda

model = Sequential()

model.add(Lambda(lambda x: (x / 255.0) - 0.5,
	input_shape=(160,320,3)))

model.add(Flatten())

model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('relu'))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, validation_split=0.2, 
	shuffle=True, nb_epoch=7)

model.save('model.h5')
exit()
