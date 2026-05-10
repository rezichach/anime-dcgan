from pathlib import Path
import math

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import tensorflow as tf

from models import NOISE_DIM, build_generator


# =========================
# Settings
# =========================

EPOCH_TO_USE = 50          # Set to None to use latest checkpoint
NUM_IMAGES = 16

BASE_DIR = Path(__file__).resolve().parents[1]

CHECKPOINTS_DIR = BASE_DIR / "outputs" / "checkpoints"
OUTPUT_DIR = BASE_DIR / "outputs" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# Find generator checkpoint
# =========================

if EPOCH_TO_USE is not None:
    GENERATOR_PATH = CHECKPOINTS_DIR / f"generator_epoch_{EPOCH_TO_USE:03d}.weights.h5"

    if not GENERATOR_PATH.exists():
        raise FileNotFoundError(
            f"Generator checkpoint not found:\n{GENERATOR_PATH}\n\n"
            "Check which files exist in outputs/checkpoints."
        )
else:
    checkpoint_files = sorted(CHECKPOINTS_DIR.glob("generator_epoch_*.weights.h5"))

    if len(checkpoint_files) == 0:
        raise FileNotFoundError(
            f"No generator checkpoints found in:\n{CHECKPOINTS_DIR}\n\n"
            "Train first with: python src/train.py"
        )

    GENERATOR_PATH = checkpoint_files[-1]


# =========================
# Load generator weights
# =========================

print(f"Loading generator weights from: {GENERATOR_PATH}")

generator = build_generator()
generator.load_weights(str(GENERATOR_PATH))

print("Generator loaded successfully.")


# =========================
# Generate new images
# =========================

noise = tf.random.normal([NUM_IMAGES, NOISE_DIM])
generated_images = generator(noise, training=False)

# Convert from [-1, 1] to [0, 1]
generated_images = (generated_images + 1) / 2
generated_images = tf.clip_by_value(generated_images, 0, 1)

generated_images = generated_images.numpy()


# =========================
# Save individual images
# =========================

for i in range(NUM_IMAGES):
    image_path = OUTPUT_DIR / f"generated_{i + 1:03d}.png"

    plt.figure(figsize=(2, 2))
    plt.imshow(generated_images[i])
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(image_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"Saved: {image_path}")


# =========================
# Save grid image
# =========================

grid_size = math.ceil(math.sqrt(NUM_IMAGES))

plt.figure(figsize=(grid_size * 2, grid_size * 2))

for i in range(NUM_IMAGES):
    plt.subplot(grid_size, grid_size, i + 1)
    plt.imshow(generated_images[i])
    plt.axis("off")

if EPOCH_TO_USE is not None:
    grid_path = OUTPUT_DIR / f"generated_grid_epoch_{EPOCH_TO_USE:03d}.png"
else:
    grid_path = OUTPUT_DIR / "generated_grid_latest.png"

plt.tight_layout()
plt.savefig(grid_path)
plt.close()

print(f"Saved grid: {grid_path}")
print("Done.")