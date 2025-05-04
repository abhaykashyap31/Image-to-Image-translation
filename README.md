
# 📘 Image-to-Image Translation with PyTorch

This README provides a detailed overview of `pics2pics.ipynb`, a Jupyter notebook that demonstrates an **image-to-image translation** workflow using PyTorch and the **facades dataset**. Inspired by the [pix2pix framework](https://arxiv.org/abs/1611.07004), this notebook focuses on preprocessing, custom dataset creation, and visualization.

---

## 🧠 Project Summary

This notebook showcases the data preparation pipeline for image-to-image translation using paired images of architectural facades and their semantic labels. The workflow is particularly tailored for models like pix2pix (a type of conditional GAN), but can be extended to any supervised image translation task.

---

## ✨ Key Features

* **📦 Automatic Dataset Handling**
  Downloads and extracts the [facades dataset](http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/facades.tar.gz) automatically.

* **🧰 Custom PyTorch Dataset Class**
  `FacadeDataset` efficiently loads and splits paired images into input-label and target-photo halves.

* **🧼 Data Preprocessing**
  Images are resized to 256x256, normalized, and converted to tensors.

* **📤 Data Loaders**
  Uses PyTorch’s `DataLoader` to efficiently load train, validation, and test splits.

* **🖼️ Visualization**
  Provides a batch visualization function to inspect input-target pairs before training.

---

## 📂 Notebook Structure

### 1. **Setup and Imports**

* Sets up device (CPU/GPU) and imports essential libraries: `torch`, `torchvision`, `PIL`, `matplotlib`, `tqdm`, and more.

### 2. **Dataset Acquisition**

* Downloads the facades dataset using `wget`.
* Extracts it into directories: `train/`, `val/`, and `test/`.

### 3. **Custom Dataset Class**

* Defines `FacadeDataset` to:

  * Load paired images.
  * Split each image into left (label) and right (facade).
  * Apply transformations to each part.

### 4. **Data Preparation**

* Sets up image transformation pipeline.
* Instantiates `Dataset` and `DataLoader` for training, validation, and testing.

### 5. **Visualization**

* `show_train_batch()` displays a batch of paired label-photo images using `matplotlib`.

---

## ▶️ Getting Started

1. **Run the notebook** in a local Jupyter environment or on **Google Colab** with GPU support.
2. **Execute the dataset download cell** – the facades dataset will be automatically downloaded and extracted.
3. **Visualize samples** using the provided function to verify preprocessing.

> **Note:** Model architecture, training loop, and inference code are **not included** — this notebook focuses solely on the data pipeline.

---

## 🛠️ Customization Tips

* **Using a different dataset?**
  Replace the dataset URL and modify the `FacadeDataset` class if your image pairing format differs.

* **Change input image size:**
  Modify `transforms.Resize((256, 256))` in the transform pipeline.

* **Adjust batch size:**
  Set the `batch_size` in the `DataLoader` depending on your available hardware.

---

## 📦 Dependencies

Install required packages with:

```bash
pip install torch torchvision pillow matplotlib tqdm
```

### Required Libraries:

* Python 3.x
* PyTorch
* torchvision
* Pillow (PIL)
* matplotlib
* tqdm

---

## 🙏 Acknowledgments

* Dataset and concept adapted from the [Berkeley AI Research Lab](http://efrosgans.eecs.berkeley.edu/pix2pix/).
* Inspired by the original [pix2pix paper (Isola et al., 2016)](https://arxiv.org/abs/1611.07004).

---

