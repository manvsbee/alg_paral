from mpi4py import MPI
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Split the training data across MPI processes
def split_data(data, num_processes, rank):
    num_samples = len(data)
    chunk_size = num_samples // num_processes
    start = rank * chunk_size
    end = (rank + 1) * chunk_size if rank < num_processes - 1 else num_samples
    return data[start:end]

# Preprocess the local data
def preprocess_data(data):
    return data.astype('float32') / 255.0

# Define the local model
def create_model():
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
        keras.layers.Conv2D(32, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Dropout(0.25),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

# Train the local model
def train_model(model, x_train, y_train):
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train, y_train, epochs=10, batch_size=64)

# Reduce gradients and synchronize model parameters
def reduce_gradients(model):
    grads = [grad.numpy() for grad in model.trainable_weights]
    reduced_grads = [np.zeros_like(grad) for grad in grads]
    for i in range(len(grads)):
        comm.Allreduce(grads[i], reduced_grads[i], op=MPI.SUM)
    grads = [tf.convert_to_tensor(grad) for grad in reduced_grads]
    model.optimizer.apply_gradients(zip(grads, model.trainable_weights))


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    num_processes = comm.Get_size()
    rank = comm.Get_rank()

    # Load the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

    # Split and preprocess the local training data
    x_train_local = split_data(x_train, num_processes, rank)
    y_train_local = split_data(y_train, num_processes, rank)
    x_train_local = preprocess_data(x_train_local)

    # Create the local model
    model = create_model()

    # Train the local model
    train_model(model, x_train_local, y_train_local)

    # Reduce gradients and synchronize model parameters
    reduce_gradients(model)

    # Evaluate the model
    x_test_local = preprocess_data(x_test)
    test_loss, test_acc = model.evaluate(x_test_local, y_test)
    print("Process %d - Test accuracy: %.4f" % (rank, test_acc))
