# WGAN-GP on 50k Celeba Dataset

This repository contains the implementation of a Wasserstein GAN with Gradient Penalty (WGAN-GP) trained on the CelebA dataset. The model is designed to generate 64x64 images of celebrity faces, using a deep convolutional architecture for both the generator and the critic.

## Directory Structure

- **checkpoints/**: Contains the saved model checkpoints during training.
- **output/**: Stores the generated images at every training epochs.
- **critic.py**: Defines the architecture of the critic network.
- **dataset_loader.py**: Handles loading and preprocessing of the CelebA dataset.
- **generator.py**: Defines the architecture of the generator network.
- **image_saver.py**: Utility to save generated images during training.
- **showcase.png**: A sample image showcasing the results of the model after training.
- **train_wdcgan.py**: Script to train the WGAN-GP model.

## Prerequisites

- Python 3.9-3.11
- PyTorch
- Matplotlib
- torchvision
- tqdm

## Dataset
The CelebA dataset is not included in this repository. You can download the dataset from [Kaggle](https://www.kaggle.com/datasets/therealcyberlord/50k-celeba-dataset-64x64). Once downloaded, ensure that the images are resized to 64x64 pixels before training.

## Training the Model

To train the WGAN-GP model, run the following command:

```bash
python train_wdcgan.py
```

You can modify the hyperparameters like `N_EPOCHS`, `BATCH_SIZE`, and `C_LAMBDA` according to your needs.

## Outputs

- **checkpoints/**: The model checkpoints are saved here for every few epochs.
- **output/**: This directory contains the generated images at various stages of training.

## Showcase

Below is a sample output from the trained model:

![gen_80](https://github.com/user-attachments/assets/716277bc-6861-4cfd-bdf4-1dc7d77221cf)

## Acknowledgments

This implementation is based on the original WGAN-GP paper: ["Improved Training of Wasserstein GANs"](https://arxiv.org/abs/1704.00028).

## License

This project is licensed under the MIT License.
