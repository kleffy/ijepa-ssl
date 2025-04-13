# I-JEPA Animal Classifier App

This simple Streamlit application provides an interactive interface to visualize and explore the animal embedding space created by the I-JEPA self-supervised learning model.

## Features

- Interactive visualization of the feature space using t-SNE
- Upload and classify new animal images
- Find and display nearest neighbors from the dataset
- Visualize where new images fit in the embedding space

## Installation

```bash
# Install Streamlit and other dependencies
pip install -r requirements-app.txt
```

## Usage

1. Make sure the model is trained and saved in the expected location
2. Run the Streamlit app:

```bash
streamlit run app.py
```

3. Open your browser to the URL displayed in the terminal (typically http://localhost:8501)

## Directory Structure

The app expects the following structure:

```
ijepa-ssl/
├── app.py                   # Streamlit application code
├── data/                    # Dataset directory
│   ├── Chinchilla/          # Class directories with images
│   ├── Hamster/
│   └── Rabbit/
└── experiments/
    └── ijepa_experiment_V8/ # Experiment output directory
        └── ijepa_model.pth  # Trained model
```

## Troubleshooting

- If the model fails to load, check that the path in `MODEL_PATH` variable matches your actual model location
- Make sure the dataset structure follows the expected format

## Customization

You can customize the app by modifying these variables at the top of the script:

- `MODEL_PATH`: Path to your trained model
- `IMAGE_SIZE`: Size to resize images (should match model training size)
- `DATA_DIR`: Directory containing your animal images