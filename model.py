from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Add, GlobalAveragePooling2D, Dense
)
from tensorflow.keras.utils import to_categorical

# Load and preprocess Fashion MNIST dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# Normalize and add channel dimension
x_train = x_train[..., None] / 255.0
x_test = x_test[..., None] / 255.0

# One-hot encode the labels
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# Build the ResNet model
inputs = Input(shape=(28, 28, 1))

# Initial convolution and pooling
x = Conv2D(64, (7, 7), strides=2, padding="same", activation=None)(inputs)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPooling2D((3, 3), strides=2, padding="same")(x)

# First residual block
shortcut = x
x = Conv2D(64, (3, 3), padding="same", activation=None)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2D(64, (3, 3), padding="same", activation=None)(x)
x = BatchNormalization()(x)
x = Add()([x, shortcut])
x = ReLU()(x)

# Second residual block (with downsampling)
shortcut = Conv2D(128, (1, 1), strides=2, padding="same", activation=None)(x)
shortcut = BatchNormalization()(shortcut)
x = Conv2D(128, (3, 3), strides=2, padding="same", activation=None)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2D(128, (3, 3), padding="same", activation=None)(x)
x = BatchNormalization()(x)
x = Add()([x, shortcut])
x = ReLU()(x)

# Third residual block (with downsampling)
shortcut = Conv2D(256, (1, 1), strides=2, padding="same", activation=None)(x)
shortcut = BatchNormalization()(shortcut)
x = Conv2D(256, (3, 3), strides=2, padding="same", activation=None)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2D(256, (3, 3), padding="same", activation=None)(x)
x = BatchNormalization()(x)
x = Add()([x, shortcut])
x = ReLU()(x)

# Fourth residual block (with downsampling)
shortcut = Conv2D(512, (1, 1), strides=2, padding="same", activation=None)(x)
shortcut = BatchNormalization()(shortcut)
x = Conv2D(512, (3, 3), strides=2, padding="same", activation=None)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = Conv2D(512, (3, 3), padding="same", activation=None)(x)
x = BatchNormalization()(x)
x = Add()([x, shortcut])
x = ReLU()(x)

# Global pooling and output
x = GlobalAveragePooling2D()(x)
outputs = Dense(10, activation="softmax")(x)

# Compile the model
model = Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")
