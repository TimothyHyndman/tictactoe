"""
Deep Q learning AI for tic-tac-toe
"""
import tensorflow as tf


def main():
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(10),
        tf.keras.layers.Dense(64, activation='sigmoid'),
        tf.keras.layers.Dense(1)
    ])

    print(model)


if __name__ == '__main__':
    main()
