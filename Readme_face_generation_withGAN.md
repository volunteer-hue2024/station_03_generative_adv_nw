This notebook implements a **Deep Convolutional Generative Adversarial Network (DCGAN)** to generate human faces using the CelebA dataset. It leverages TensorFlow and multi-GPU distribution strategies.

---

## 1. Data Loading and Preprocessing

This block handles the ingestion of images from the `celeba-dataset`.

* **Normalization:** Images are resized to **128x128** and normalized to a range of **[-1, 1]**. This is crucial because the Generator uses a `tanh` activation function.
* **Memory Management:** It loads the first 10,000 images into memory and utilizes `tf.data.Dataset` for efficient shuffling and batching.
* **Prefetching:** Uses `tf.data.AUTOTUNE` to prepare subsequent batches while the GPU is processing the current one, reducing bottlenecking.

---

## 2. Model Architecture (Generator & Discriminator)

This block defines the two "players" in the adversarial game:

### The Generator

* **Purpose:** Learns to map a 100-dimensional random noise vector to a realistic 128x128x3 image.
* **Layers:** Starts with a `Dense` layer and uses **Conv2DTranspose** (fractionally-strided convolutions) to upsample the spatial dimensions from 8x8 → 16x16 → 32x32 → 64x64 → 128x128.
* **Refinements:** Uses `BatchNormalization` to stabilize training and `LeakyReLU` as the activation function.

### The Discriminator

* **Purpose:** A binary classifier that learns to distinguish between "real" images (from the dataset) and "fake" images (from the generator).
* **Layers:** Uses standard `Conv2D` layers with strides to downsample the image.
* **Refinements:** Uses `Dropout` (0.3) to prevent the discriminator from becoming too powerful too quickly, which would stop the generator from learning.

---

## 3. Loss Functions and Optimizers

* **Loss:** Both models use **Binary Cross-Entropy**.
* **Generator Loss:** Penalizes the model if the discriminator correctly identifies its fake images as 0. It wants the discriminator to output 1s (Real).
* **Discriminator Loss:** Sums the loss for real images (should be 1) and fake images (should be 0).


* **Optimizer:** **Adam** is used for both, with a learning rate of `1e-4`.

---

## 4. Training Step and Visualization

* **`train_step`:** Wrapped in a `tf.GradientTape`, it calculates gradients for both models simultaneously. It updates weights based on how well the generator fooled the discriminator and how well the discriminator caught the generator.
* **`generate_and_show_images`:** A utility function that takes a "fixed noise" seed and plots a 4x4 grid of images. This allows you to visually track the generator's progress after every epoch.

---

## 5. Multi-GPU Distribution Strategy

The notebook uses `tf.distribute.MirroredStrategy` to speed up training.

* **Strategy Scope:** The models and optimizers are defined inside `strategy.scope()` so their variables are mirrored across available GPUs (e.g., Tesla T4s).
* **Distributed Dataset:** The standard dataset is converted into a distributed version to feed both GPUs in parallel.
* **`distributed_train_step`:** This executes the training step on each replica and aggregates (reduces) the loss values.

---

## 6. Execution and Saving

The final block initiates the training loop for **100 epochs**. Once finished, it saves the `generator` model to a file named `generator_800.h5`. This saved model can later be used to generate faces without needing to retrain or keep the discriminator.

---

# Part 2

**How to load that saved `.h5` model for future use?**

To load and use your saved GAN generator, you need to reconstruct the model's environment so it understands how to interpret the `.h5` file. Since your training was performed within a `MirroredStrategy` scope, it is best practice to load it similarly if you intend to continue training, or simply load it as a standard Keras model for generation.

Here is the blockwise breakdown of how to load the model and generate new images:

## 1. Environment Setup

First, ensure you have the same dependencies imported as they were during training.

```python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Use the same constant defined in your notebook
NOISE_DIM = 100 

```

---

## 2. Loading the Model

You can use `tf.keras.models.load_model` to bring the generator back to life.

* **For Inference (Generating Images):** You can load it directly.
* **For Continued Training:** If you used custom objects or specific distribution strategies, you would wrap this in a strategy scope.

```python
# Load the saved generator
generator = tf.keras.models.load_model('generator_800.h5')

# Optional: Verify the architecture
generator.summary()

```

---

## 3. Generating New Faces

To use the model, you must provide it with "latent vectors" (random noise) of the same shape it was trained on ().

```python
def generate_new_faces(num_images=16):
    # 1. Create random noise seeds
    noise = tf.random.normal([num_images, NOISE_DIM])
    
    # 2. Predict (generate) images
    # training=False is important for BatchNormalization layers
    predictions = generator(noise, training=False)
    
    # 3. Post-process: Rescale from [-1, 1] back to [0, 1]
    predictions = (predictions + 1) / 2.0
    
    # 4. Visualize
    fig = plt.figure(figsize=(6, 6))
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i])
        plt.axis('off')
    plt.show()

generate_new_faces()

```

---

## Important Considerations

* **Activation Range:** Remember that your generator's final layer uses a `tanh` activation, outputting values between -1 and 1. Always rescale these values back to the 0–1 range for `matplotlib` or 0–255 for standard image saving.
* **Missing Weights:** If you only saved the weights (using `.save_weights`), you would need to call `build_generator()` first to initialize the architecture before running `model.load_weights()`. Since you used `.save()`, the entire architecture and optimizer state are included in the file.

**How to export these generated images as actual PNG or JPEG files to your local drive?**

To export your generated images as files (like `.png` or `.jpg`), you can use the `PIL` (Pillow) library or `matplotlib`. Using `PIL` is generally better for saving individual high-quality images.

Here is the blockwise code to generate and save your results:

## 1. Exporting Individual Images

This block converts the Generator's output from a TensorFlow tensor into a standard image file on your disk.

```python
import os
from PIL import Image

def save_generated_faces(num_images=5, output_dir='generated_results'):
    # Create directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Generate noise and images
    noise = tf.random.normal([num_images, NOISE_DIM])
    predictions = generator(noise, training=False) #

    # 2. Rescale to [0, 255] and convert to uint8
    # We shift from [-1, 1] to [0, 1], then scale to byte range
    predictions = (predictions + 1) / 2.0 
    predictions = (predictions.numpy() * 255).astype(np.uint8)

    # 3. Save each image using PIL
    for i in range(num_images):
        img = Image.fromarray(predictions[i])
        img.save(f"{output_dir}/face_{i+1}.png")
    
    print(f"Successfully saved {num_images} images to '{output_dir}'")

save_generated_faces(10)

```

---

## 2. Creating a Comparison Grid

If you want to save a single image containing a "contact sheet" of many faces (similar to the visualization in your notebook), use this block:

```python
def save_image_grid(rows=4, cols=4, filename='face_grid.png'):
    num_images = rows * cols
    noise = tf.random.normal([num_images, NOISE_DIM])
    predictions = generator(noise, training=False)
    predictions = (predictions + 1) / 2.0

    fig, axes = plt.subplots(rows, cols, figsize=(rows*2, cols*2))
    idx = 0
    for r in range(rows):
        for c in range(cols):
            axes[r, c].imshow(predictions[idx])
            axes[r, c].axis('off')
            idx += 1
    
    # Save the entire figure
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Grid saved as {filename}")

save_image_grid(4, 4)

```

---

## Key Technical Details

* **Data Type Conversion:** Neural networks output `float32` values. Image files require `uint8` (integers from 0 to 255). The step `(predictions * 255).astype(np.uint8)` is critical for a valid image file.
* **Batch Normalization:** When saving, we keep `training=False` because the Generator uses `BatchNormalization` layers. This ensures the model uses its "learned" moving averages rather than the statistics of the current small batch.
* **Output Path:** If you are running this on Kaggle or Colab, these files will appear in the "Files" pane on the left-hand sidebar.

**How to download these files automatically to your local computer if you are using a cloud environment like Google Colab?**
