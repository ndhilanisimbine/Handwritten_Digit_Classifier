# Import TensorFlow 2.0
import tensorflow as tf

# MIT introduction to deep learning package
!pip install mitdeeplearning --quiet
import mitdeeplearning as mdl

# Other packages
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm

# Checking if GPU is available
assert len(tf.config.list_physical_devices('GPU')) > 0, "No GPU found! Please enable GPU in your runtime."

# Loading the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalizing the images
train_images = (np.expand_dims(train_images, axis=-1) / 255.0).astype(np.float32)
train_labels = train_labels.astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1) / 255.0).astype(np.float32)
test_labels = test_labels.astype(np.int64)

# Visualizing the MNIST dataset
plt.figure(figsize=(10, 10))
random_inds = np.random.choice(60000, 36)
for i in range(36):
    plt.subplot(6, 6, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(train_images[image_ind]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[image_ind])
plt.show()



def build_fc_model():
    fc_model = tf.keras.Sequential([
        # First define a Flatten layer
        tf.keras.layers.Flatten(),

        # Define the first Dense layer with 128 units and ReLU activation
        tf.keras.layers.Dense(128, activation='relu'),

        # Define the second Dense layer with 10 units (for 10 classes) and softmax activation
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return fc_model

# Building the model
model = build_fc_model()


# Compiling the model
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Defining batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 5

# Training the model
model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)

def build_cnn_model():
    cnn_model = tf.keras.Sequential([
        # First convolutional layer: 32 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        
        # First max pooling layer: 2x2 pool size
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Second convolutional layer: 64 filters, 3x3 kernel, ReLU activation
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        
        # Second max pooling layer: 2x2 pool size
        tf.keras.layers.MaxPooling2D((2, 2)),
        
        # Flatten the feature maps to a 1D vector
        tf.keras.layers.Flatten(),
        
        # Fully connected layer with 128 units and ReLU activation
        tf.keras.layers.Dense(128, activation=tf.nn.relu),
        
        # Output layer with 10 units (one per class) and softmax activation
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return cnn_model

# Building the CNN model
cnn_model = build_cnn_model()

# Printing the model summary to verify the architecture
print(cnn_model.summary())



cnn_model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Defining the batch size and number of epochs
BATCH_SIZE = 64
EPOCHS = 5

# Training the CNN model
cnn_model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)


# Evaluating the CNN model on the test dataset
test_loss, test_acc = cnn_model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)


# Making predictions on the test dataset
predictions = cnn_model.predict(test_images)


# Displaying the prediction for the first image in the test set
print(predictions[0])

# Identifying the digit with the highest confidence prediction for the first image
predicted_digit = np.argmax(predictions[0])
print("Predicted digit for the first image:", predicted_digit)

# Checking if the prediction is correct by comparing it to the true label
print("True label for the first image:", test_labels[0])


# Displaying the first image and its predicted and true labels
plt.imshow(test_images[0].reshape(28, 28), cmap=plt.cm.binary)
plt.title(f"Predicted: {predicted_digit}, True: {test_labels[0]}")
plt.show()


# Plotting the first X test images, their predicted label, and the true label
num_rows = 5
num_cols = 4
num_images = num_rows * num_cols
plt.figure(figsize=(2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, num_cols, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].reshape(28, 28), cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    plt.xlabel(f"Pred: {predicted_label}\nTrue: {true_label}", color=color)
plt.tight_layout()
plt.show()


# Feeding the images into the model and obtain the predictions
with tf.GradientTape() as tape:
    logits = cnn_model(images, training=True)  # Forward pass


# Computing the categorical cross entropy loss
loss_value = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)



# Computing the gradients with respect to the trainable variables
grads = tape.gradient(loss_value, cnn_model.trainable_variables)

# Applying the gradients using the optimizer
optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))



# Rebuilding the CNN model
cnn_model = build_cnn_model()

batch_size = 12
optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

# Training loop using GradientTape
for idx in tqdm(range(0, train_images.shape[0], batch_size)):
    # Grab a batch of training data
    images, labels = train_images[idx:idx+batch_size], train_labels[idx:idx+batch_size]
    images = tf.convert_to_tensor(images, dtype=tf.float32)

    # GradientTape to record differentiation operations
    with tf.GradientTape() as tape:
        # Forward pass: feed the images into the model and obtain the predictions
        logits = cnn_model(images, training=True)
        
        # Computing the categorical cross entropy loss
        loss_value = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
    
    # Computing the gradients of the loss w.r.t the trainable variables
    grads = tape.gradient(loss_value, cnn_model.trainable_variables)
    
    # Applying gradients using the optimizer
    optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))

    # Log the loss for each step (optional visualization if you want to track loss)
    print(f"Batch {idx}, Loss: {tf.reduce_mean(loss_value).numpy()}")




