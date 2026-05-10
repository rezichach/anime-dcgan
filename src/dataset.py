import os
import tensorflow as tf
import matplotlib.pyplot as plt

IMAGE_SIZE = 64
BATCH_SIZE = 32

DATASET_PATH = r"D:\anime-dcgan\data\anime_faces"


def load_and_preprocess_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)

    # TensorFlow needs the shape after decode_image
    image.set_shape([None, None, 3])

    # Resize every image to 64x64
    image = tf.image.resize(image, [IMAGE_SIZE, IMAGE_SIZE])

    # Convert pixel values from [0, 255] to [-1, 1]
    image = tf.cast(image, tf.float32)
    image = (image - 127.5) / 127.5

    return image


def create_dataset(dataset_path, batch_size):
    image_paths = []

    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    for root, dirs, files in os.walk(dataset_path):
        for file_name in files:
            if file_name.lower().endswith(valid_extensions):
                image_paths.append(os.path.join(root, file_name))

    print(f"Found {len(image_paths)} images.")

    if len(image_paths) == 0:
        raise ValueError("No images found. Check your DATASET_PATH.")

    dataset = tf.data.Dataset.from_tensor_slices(image_paths)
    dataset = dataset.shuffle(buffer_size=len(image_paths))
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


def show_real_images(dataset):
    # Take one batch from the dataset
    batch = next(iter(dataset))

    plt.figure(figsize=(6, 6))

    for i in range(16):
        plt.subplot(4, 4, i + 1)

        # Convert from [-1, 1] back to [0, 1] for displaying
        image = (batch[i] + 1) / 2

        plt.imshow(image)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    dataset = create_dataset(DATASET_PATH, BATCH_SIZE)
    show_real_images(dataset)