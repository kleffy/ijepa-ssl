import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import logging
import os

from utils.utils import get_dataloader
from utils.embeddings import generate_embeddings

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP


def create_evaluation_summary(results, output_dir, config):
    """Create and save evaluation summary plots and CSV"""
    summary_df = pd.DataFrame(list(results.items()), columns=['Metric', 'Value'])
    plt.figure(figsize=(10, 12))

    # Few-shot learning plot
    shot_metrics = {k: v for k, v in results.items() if 'shot_accuracy' in k}
    if shot_metrics:
        shots_data = sorted([(int(k.split('_')[0]), v) for k, v in shot_metrics.items()])
        shots = [item[0] for item in shots_data]
        accuracies = [item[1] for item in shots_data]
        ax1 = plt.subplot(2, 1, 1)
        ax1.plot(shots, accuracies, 'o-', label='Few-Shot Accuracy')
        ax1.set_title(f'Few-Shot Learning Performance')
        ax1.set_xlabel('Number of Shots per Class (k)')
        ax1.set_ylabel('Average Accuracy')
        ax1.set_xticks(shots)
        ax1.grid(alpha=0.3)
        ax1.legend()

    # Main metrics bar chart
    main_metrics_keys = ['linear_eval_accuracy', 'nearest_neighbor_accuracy', 'nmi', 'ari']
    main_metrics_labels = ['Linear Eval', 'k-NN Eval', 'NMI (Cluster)', 'ARI (Cluster)']
    main_values = [results.get(m, 0) for m in main_metrics_keys]
    ax2 = plt.subplot(2, 1, 2)
    bars = ax2.bar(main_metrics_labels, main_values, color=['skyblue', 'lightgreen', 'salmon', 'gold'])
    ax2.set_title(f'Overall Evaluation Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1.05)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f'{yval:.3f}', va='bottom', ha='center')

    plt.xticks(rotation=15, ha='right')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    plot_path = os.path.join(output_dir, 'evaluation_results_plot.png')
    csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
    plt.savefig(plot_path)
    summary_df.to_csv(csv_path, index=False)
    plt.close()

    logging.info(f"Evaluation plot saved to {plot_path}")
    logging.info(f"Evaluation summary saved to {csv_path}")


def visualize_features(encoder, config, class_names):
    """Extract features and visualize them using UMAP/t-SNE"""
    vis_reduction = config['vis_reduction']
    if UMAP is None and vis_reduction.upper() == 'UMAP':
        logging.warning("UMAP not installed. Falling back to t-SNE.")
        vis_reduction = 'TSNE'

    logging.info("Starting Feature Space Visualization")
    vis_dataloader, _, _, _ = get_dataloader(config, shuffle=False)
    if vis_dataloader is None:
        logging.error("Failed visualization dataloader.")
        return

    features, labels = generate_embeddings(encoder, vis_dataloader, config)
    if features is None:
        logging.error("Failed embeddings generation for visualization.")
        return

    logging.info(f"Performing dimensionality reduction using {vis_reduction}...")
    n_samples = features.shape[0]
    n_neighbors = min(15, n_samples - 1) if n_samples > 1 else 1
    perplexity = min(30, n_samples - 1) if n_samples > 1 else 1
    if n_samples <= 1:
        logging.warning("Warning: Only <= 1 sample found.")
        return

    try:
        features_scaled = StandardScaler().fit_transform(features)
        if vis_reduction.upper() == 'UMAP':
            if UMAP is None: raise ImportError("UMAP not available")
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)
        elif vis_reduction.upper() == 'TSNE':
            reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_iter=1000, init='pca', learning_rate='auto')
        else:
            if UMAP is None: raise ImportError("UMAP not available")
            reducer = UMAP(n_components=2, random_state=42, n_neighbors=n_neighbors, min_dist=0.1)

        features_2d = reducer.fit_transform(features_scaled)
        logging.info("Dimensionality reduction complete.")

    except ImportError as e:
        logging.error(f"Error during DR import: {e}.")
        return
    except Exception as e:
        logging.error(f"Error during {vis_reduction} fitting: {e}")
        return

    plt.figure(figsize=(12, 10))
    unique_labels = np.unique(labels)
    num_classes_present = len(unique_labels)

    # Handle potential mismatch between labels and class names
    plot_class_names = class_names if class_names and len(class_names) >= max(unique_labels)+1 else [f"Class {i}" for i in unique_labels]

    # Create a mapping from label to plot index
    label_to_plot_idx = {label: i for i, label in enumerate(unique_labels)}
    plot_labels = np.array([label_to_plot_idx[l] for l in labels])

    # Create scatter plot with color coding
    cmap = plt.get_cmap('viridis', num_classes_present)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=plot_labels, cmap=cmap, s=15, alpha=0.8)

    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(i), markersize=10)
               for i in range(num_classes_present)]
    plt.legend(handles, plot_class_names, title="Classes")

    plt.title(f'Feature Space Visualization ({vis_reduction}) - I-JEPA Trained Encoder')
    plt.xlabel(f'{vis_reduction} Dimension 1')
    plt.ylabel(f'{vis_reduction} Dimension 2')
    plt.grid(True, linestyle='--', alpha=0.5)

    # Save the plot
    feature_map_path = os.path.join(config['output_dir'], 'feature_map.png')
    os.makedirs(os.path.dirname(feature_map_path), exist_ok=True)
    plt.savefig(feature_map_path)
    logging.info(f"Feature map saved to {feature_map_path}")
    plt.close()