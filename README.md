# I-JEPA: Self-Supervised Learning for Animal Classification

## Overview

This repository implements the Image-based Joint-Embedding Predictive Architecture (I-JEPA) for self-supervised learning on an animal image dataset. I-JEPA is a state-of-the-art approach that learns meaningful image representations by predicting representations of target image regions from context regions without relying on hand-crafted data augmentations.

The project also includes a comparative evaluation with SimCLR (Simple Contrastive Learning of Representations), providing insights into the strengths and weaknesses of different self-supervised learning approaches.

## Features

- Implementation of I-JEPA architecture with Vision Transformers
- Multi-block masking strategy for effective representation learning
- Comprehensive evaluation suite (linear evaluation, k-NN, clustering metrics, few-shot learning)
- Feature space visualization using t-SNE
- Comparative analysis with SimCLR

## Installation

### Using pip

```bash
# Clone the repository
git clone https://github.com/kleffy/ijepa-ssl.git
cd ijepa-ssl

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Using Docker

```bash
# Build the Docker image
docker build -t ijepa-ssl .

# Run the container
docker run --gpus all -v /path/to/dataset:/app/data ijepa-ssl
```

## Project Structure

```
ijepa-ssl/
├── app.py                  # Streamlit frontend
├── main.py                 # Main entry point for training and evaluation
├── config/                 # Configuration files
│   └── config.yaml         # Main configuration
├── models/
│   └── model.py            # I-JEPA model implementation
├── utils/                  # Utility functions
│   ├── utils.py            # General utilities
│   ├── eval.py             # Evaluation functions
│   ├── viz.py              # Visualization utilities
│   └── embeddings.py       # Embedding generation utilities
├── dataset/
│   └── animal_dataset.py   # Dataset loader
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker configuration
└── README.md               # Project documentation
```

## Usage

### Training

To train the I-JEPA model:

```bash
python main.py --config config/config.yaml
```

You can customize the training by modifying the `config.yaml` file.

### Evaluation

The evaluation is automatically performed after training. The results are saved in the output directory specified in the configuration.

```bash
#TODO: Support running only evaluation on a pretrained model
python main.py --config config/config.yaml --eval-only
```

## Dataset

The project uses an animal image dataset containing three classes:
- Chinchilla
- Hamster
- Rabbit

The dataset should be organized in the following structure:

```
data/
├── Chinchilla/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── Hamster/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── Rabbit/
    ├── image1.jpg
    ├── image2.jpg
    └── ...
```

## Model Architecture

### I-JEPA Components

1. **Context Encoder**: A Vision Transformer (ViT) that processes visible image patches.
2. **Target Encoder**: An identical network updated using an exponential moving average (EMA) of the context encoder.
3. **Predictor**: A lightweight transformer that predicts target block representations from context embeddings.

### Masking Strategy

- **Target blocks**: 4 blocks with scale 0.15-0.2 and aspect ratio 0.75-1.5
- **Context block**: A single block with scale 0.85-1.0, excluding regions overlapping with target blocks

## Evaluation Results

### I-JEPA Performance

| Metric | Score |
|--------|-------|
| Linear Evaluation Accuracy | 100.0% |
| k-NN Accuracy | 93.1% |
| Normalized Mutual Information (NMI) | 0.498 |
| Adjusted Rand Index (ARI) | 0.612 |
| 1-shot Accuracy | 60.1% |
| 5-shot Accuracy | 66.8% |
| 10-shot Accuracy | 68.0% |

### SimCLR Performance

| Metric | Score |
|--------|-------|
| Linear Evaluation Accuracy | 95.8% |
| k-NN Accuracy | 87.5% |
| Normalized Mutual Information (NMI) | 0.373 |
| Adjusted Rand Index (ARI) | 0.246 |
| 1-shot Accuracy | 63.1% |
| 5-shot Accuracy | 68.1% |
| 10-shot Accuracy | 74.3% |

## Configuration Options

Key configuration parameters in `config.yaml`:

```yaml
# Experiment Configuration
experiment_name: 'ijepa_experiment'
version: 'V8'
seed: 429
output_dir: '/path/to/output'

# Dataset
dataset_path: '/path/to/data'
image_size: 224
patch_size: 16
validation_ratio: 0.1

# Model
model_name: 'vit_base_patch16_224'
pretrained: true
embed_dim: 768
predictor_depth: 6
predictor_embed_dim: 768

# Masking
mask_scales:
  - ratio: 0.85
    aspect_ratio_range: [0.75, 1.5]
  - ratio: 0.15
    aspect_ratio_range: [0.75, 1.5]
num_target_masks: 4
mask_generation_retries: 20

# Training
device: 'cuda'
batch_size: 64
epochs: 50
learning_rate: 0.0001
weight_decay: 0.05
ema_decay: 0.996
early_stopping_patience: 15

# Evaluation
eval_batch_size: 128
eval_linear_train_ratio: 0.8
eval_knn_neighbors: 20
eval_few_shot_shots: [1, 5, 10]
eval_few_shot_iterations: 20

# Visualization
vis_reduction: 'TSNE'
```

## Dependencies

- Python 3.8+
- PyTorch 1.10.0+
- torchvision 0.11.0+
- timm 0.6.0+
- scikit-learn 1.0.0+
- matplotlib 3.5.0+
- pandas 1.3.0+
- tqdm 4.62.0+
- umap-learn 0.5.2+

See `requirements.txt` for a complete list of dependencies.

## Interactive Demo

This project includes an interactive Streamlit application that demonstrates the capabilities of the trained I-JEPA model:

### Features
- Interactive visualization of the feature space using t-SNE
- Upload and classify new animal images
- Find and display nearest neighbors from the dataset
- Generate downloadable classification reports

### Running the Demo
```bash
# Install required dependencies
pip install -r requirements-app.txt

# Launch the application
streamlit run app.py
```

### Screenshots
We did a quick test with random images from the internet.
1. Chinchila image

![Random Chinchila image from the internet](/output/c_test_01.jpg) ![Test Results](/output/classification_result.png)

2. Rabbit image

![Random Rabbit image from the internet](/output/rabbit_01.jpg) ![Test Results](/output/classification_result_rabbit.png)


For full documentation of the demo application, see [APP_README.md](APP_README.md).

## References

1. Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., & Ballas, N. (2023). Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture.

2. Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Thanks to <Redacted> for providing the animal dataset and challenge structure
- Implementation based on insights from the original I-JEPA paper


