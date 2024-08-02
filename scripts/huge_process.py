import tensorflow as tf


def intensive_tf_process():
    # Create a simple model and perform some intensive computation
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(1024,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')

    # Generate random data
    x_train = tf.random.uniform((10000, 1024))
    y_train = tf.random.uniform((10000, 10))

    print("Starting intensive training process...")
    model.fit(x_train, y_train, epochs=5, batch_size=32)
    print("Intensive training process completed.")

if __name__ == "__main__":
    intensive_tf_process()