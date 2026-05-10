import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from dataset import create_dataset
from models import build_generator, build_discriminator, NOISE_DIM


# =========================
# Basic settings
# =========================

IMAGE_SIZE = 64
BATCH_SIZE = 128
EPOCHS = 50

# Smoke test mode:
# Use 100 batches first so we do not wait forever.
# Later, set this to None for full epochs.
MAX_BATCHES_PER_EPOCH = None

LEARNING_RATE = 0.0002
BETA_1 = 0.5

NUM_EXAMPLES_TO_GENERATE = 16


# =========================
# Project paths
# =========================

BASE_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = BASE_DIR / "data" / "anime_faces"

OUTPUTS_DIR = BASE_DIR / "outputs"
SAMPLES_DIR = OUTPUTS_DIR / "samples"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
GRAPHS_DIR = OUTPUTS_DIR / "graphs"

SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
GRAPHS_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Loss functions
# =========================

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def generator_loss(fake_output):
    """
    Generator wants fake images to be classified as real.
    So target labels are 1.
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    """
    Discriminator wants:
    real images -> 1
    fake images -> 0
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss


# =========================
# Save generated images
# =========================

def save_generated_images(generator, epoch, seed):
    generated_images = generator(seed, training=False)

    # Convert from [-1, 1] to [0, 1]
    generated_images = (generated_images + 1) / 2
    generated_images = tf.clip_by_value(generated_images, 0, 1)

    plt.figure(figsize=(6, 6))

    for i in range(NUM_EXAMPLES_TO_GENERATE):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i])
        plt.axis("off")

    output_path = SAMPLES_DIR / f"epoch_{epoch:03d}.png"

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved generated image grid to: {output_path}")


def save_loss_graph(generator_losses, discriminator_losses):
    plt.figure(figsize=(8, 5))
    plt.plot(generator_losses, label="Generator loss")
    plt.plot(discriminator_losses, label="Discriminator loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("GAN Training Loss")
    plt.legend()
    plt.grid(True)

    output_path = GRAPHS_DIR / "loss_curve.png"

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Saved loss graph to: {output_path}")


# =========================
# One training step
# =========================

def train_step(real_images, generator, discriminator, generator_optimizer, discriminator_optimizer):
    current_batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([current_batch_size, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        fake_images = generator(noise, training=True)

        real_output = discriminator(real_images, training=True)
        fake_output = discriminator(fake_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    generator_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(generator_gradients, generator.trainable_variables)
    )

    discriminator_optimizer.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss


# =========================
# Training loop
# =========================

def train():
    print("Loading dataset...")
    dataset = create_dataset(str(DATASET_PATH), BATCH_SIZE)

    print("Building models...")
    generator = build_generator()
    discriminator = build_discriminator()

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )

    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE,
        beta_1=BETA_1
    )

    fixed_seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

    generator_losses = []
    discriminator_losses = []

    print("\nStarting training...")
    print(f"Epochs: {EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max batches per epoch: {MAX_BATCHES_PER_EPOCH}")
    print()

    for epoch in range(1, EPOCHS + 1):
        start_time = time.time()

        epoch_gen_losses = []
        epoch_disc_losses = []

        for batch_index, real_images in enumerate(dataset, start=1):
            gen_loss, disc_loss = train_step(
                real_images,
                generator,
                discriminator,
                generator_optimizer,
                discriminator_optimizer
            )

            epoch_gen_losses.append(float(gen_loss))
            epoch_disc_losses.append(float(disc_loss))

            if batch_index % 10 == 0:
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {batch_index} | "
                    f"Gen loss: {float(gen_loss):.4f} | "
                    f"Disc loss: {float(disc_loss):.4f}"
                )

            if MAX_BATCHES_PER_EPOCH is not None and batch_index >= MAX_BATCHES_PER_EPOCH:
                break

        avg_gen_loss = float(np.mean(epoch_gen_losses))
        avg_disc_loss = float(np.mean(epoch_disc_losses))

        generator_losses.append(avg_gen_loss)
        discriminator_losses.append(avg_disc_loss)

        elapsed_time = time.time() - start_time

        print()
        print(f"Epoch {epoch}/{EPOCHS} finished")
        print(f"Average generator loss: {avg_gen_loss:.4f}")
        print(f"Average discriminator loss: {avg_disc_loss:.4f}")
        print(f"Time: {elapsed_time:.2f} seconds")
        print()

        save_generated_images(generator, epoch, fixed_seed)

        generator_save_path = CHECKPOINTS_DIR / f"generator_epoch_{epoch:03d}.weights.h5"
        generator.save_weights(str(generator_save_path))

        print(f"Saved generator weights to: {generator_save_path}")
        print("-" * 60)

    save_loss_graph(generator_losses, discriminator_losses)

    print("Training smoke test complete.")


if __name__ == "__main__":
    train()