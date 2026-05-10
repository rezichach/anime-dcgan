import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

NOISE_DIM = 100
IMAGE_SIZE = 64


def build_generator(noise_dim=NOISE_DIM):
    model = tf.keras.Sequential(name="Generator")

    model.add(layers.Input(shape=(noise_dim,)))

    model.add(layers.Dense(4 * 4 * 512, use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    model.add(layers.Reshape((4, 4, 512)))

    # 4x4 -> 8x8
    model.add(layers.Conv2DTranspose(
        256,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # 8x8 -> 16x16
    model.add(layers.Conv2DTranspose(
        128,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # 16x16 -> 32x32
    model.add(layers.Conv2DTranspose(
        64,
        kernel_size=4,
        strides=2,
        padding="same",
        use_bias=False
    ))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())

    # 32x32 -> 64x64
    model.add(layers.Conv2DTranspose(
        3,
        kernel_size=4,
        strides=2,
        padding="same",
        activation="tanh"
    ))

    return model


def build_discriminator():
    model = tf.keras.Sequential(name="Discriminator")

    model.add(layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))

    # 64x64 -> 32x32
    model.add(layers.Conv2D(
        64,
        kernel_size=4,
        strides=2,
        padding="same"
    ))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 32x32 -> 16x16
    model.add(layers.Conv2D(
        128,
        kernel_size=4,
        strides=2,
        padding="same"
    ))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 16x16 -> 8x8
    model.add(layers.Conv2D(
        256,
        kernel_size=4,
        strides=2,
        padding="same"
    ))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    # 8x8 -> 4x4
    model.add(layers.Conv2D(
        512,
        kernel_size=4,
        strides=2,
        padding="same"
    ))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())

    # One output number: real or fake
    # No sigmoid here because later we use BinaryCrossentropy(from_logits=True)
    model.add(layers.Dense(1))

    return model


def show_generated_images(generator, noise_dim=NOISE_DIM, num_images=16):
    noise = tf.random.normal([num_images, noise_dim])
    generated_images = generator(noise, training=False)

    print("Generated batch shape:", generated_images.shape)

    plt.figure(figsize=(6, 6))

    for i in range(num_images):
        plt.subplot(4, 4, i + 1)

        image = (generated_images[i] + 1) / 2
        image = tf.clip_by_value(image, 0, 1)

        plt.imshow(image)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    generator = build_generator()
    discriminator = build_discriminator()

    print("\nGENERATOR SUMMARY")
    generator.summary()

    print("\nDISCRIMINATOR SUMMARY")
    discriminator.summary()

    noise = tf.random.normal([1, NOISE_DIM])
    fake_image = generator(noise, training=False)

    print("\nSingle generated image shape:", fake_image.shape)

    fake_score = discriminator(fake_image, training=False)

    print("Discriminator output for fake image:", fake_score.numpy())
    print("Discriminator output shape:", fake_score.shape)

    show_generated_images(generator)