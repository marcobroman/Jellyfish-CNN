# %%
import os
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout


# %%
curr_dir = os.getcwd()
TRAIN_DIR = curr_dir + "/data/Train_Test_Valid/Train"
VALID_DIR = curr_dir + "/data/Train_Test_Valid/valid"
TEST_DIR = curr_dir + "/data/Train_Test_Valid/test"
BATCH_SIZE = 48
EPCOHS = 12

data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
)

train_set = data_gen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),  # Image Dimensions
    batch_size=BATCH_SIZE,
    class_mode="categorical",  # Categorical for MultiClass classification
)

valid_set = data_gen.flow_from_directory(
    VALID_DIR,
    target_size=(224, 224),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,  # turn off shuffling
)

test_set = data_gen.flow_from_directory(
    TEST_DIR,
    (224, 224),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,  # turn off shuffling
)

# %%
# Achieved 0.6499 accuracy
model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(224, 224, 3)))
model.add(MaxPooling2D((3, 3)))

model.add(Conv2D(64, (5, 5), activation="relu"))
model.add(Dropout(0.2))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (7, 7), activation="relu"))
model.add(Dropout(0.5))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(256, (3, 3), activation="relu"))
model.add(Dropout(0.3))
model.add(MaxPooling2D((2, 2)))

model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.3))

model.add(Dense(6, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",  # Monitor validation loss
    patience=3,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,  # Restore model weights from the epoch with the best value of the monitored quantity
)

# %%
model.fit(
    train_set,
    epochs=EPCOHS,
    verbose=1,
    validation_data=valid_set,
    callbacks=[early_stopping],
)

# %%
# evaluation metrics
evaluation_result = model.evaluate(test_set)

print("Test Loss:", evaluation_result[0])
print("Test Accuracy:", evaluation_result[1])

# Predictions
predictions = model.predict(test_set)

# Convert to 1-D labels
predictions = np.argmax(predictions, axis=1)

print(predictions)
print(test_set.labels)
