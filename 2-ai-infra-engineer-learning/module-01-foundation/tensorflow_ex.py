import tensorflow as tf
from tensorflow import keras

# Create model
model = keras.Sequential([
  keras.layers.Dense(50, activation='relu', input_shape=(10,)),
  keras.layers.Dense(1)
])

# Model must be compiled
model.compile(optimizer='adam', loss='mse')

# Model summary
model.summary()