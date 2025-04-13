import os
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from utils.utils import get_dataloader
from utils.viz import create_evaluation_summary
from utils.embeddings import generate_embeddings


def linear_evaluation(embeddings, labels, config):
    """Performs linear evaluation protocol"""
    logging.info("Running Linear Evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, labels, train_size=config['eval_linear_train_ratio'], stratify=labels, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    classifier = LogisticRegression(random_state=42, max_iter=1000, C=1.0)
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    logging.info(f"Linear Evaluation Accuracy: {accuracy:.4f}")
    return accuracy


def nearest_neighbor_evaluation(embeddings, labels, config):
    """Performs k-Nearest Neighbor evaluation"""
    logging.info("Running k-NN Evaluation...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    n_neighbors = min(config['eval_knn_neighbors'], len(embeddings_scaled) - 1)
    
    if n_neighbors < 1:
        logging.warning("Not enough samples for k-NN evaluation. Skipping.")
        return 0.0
    
    classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
    classifier.fit(embeddings_scaled, labels)
    y_pred = classifier.predict(embeddings_scaled)
    accuracy = accuracy_score(labels, y_pred)
    
    logging.info(f"k-NN ({n_neighbors}) Accuracy: {accuracy:.4f}")
    return accuracy


def evaluate_clustering(embeddings, labels, config, n_classes):
    """Evaluates clustering quality using KMeans"""
    logging.info("Running Clustering Evaluation (KMeans)...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    kmeans = KMeans(n_clusters=n_classes, random_state=42, n_init=10)
    cluster_preds = kmeans.fit_predict(embeddings_scaled)
    
    nmi = normalized_mutual_info_score(labels, cluster_preds)
    ari = adjusted_rand_score(labels, cluster_preds)
    
    logging.info(f"Clustering NMI: {nmi:.4f}, ARI: {ari:.4f}")
    return {'nmi': nmi, 'ari': ari}


def few_shot_evaluation(embeddings, labels, config, n_classes):
    """Evaluates few-shot learning capability"""
    logging.info("Running Few-Shot Evaluation...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)
    results = {}

    for k in config['eval_few_shot_shots']:
        logging.info(f"Evaluating {k}-shot...")
        accuracies = []
        for _ in range(config['eval_few_shot_iterations']):
            X_train_fs, y_train_fs = [], []
            X_test_fs, y_test_fs = [], []

            for c in range(n_classes):
                class_indices = np.where(labels == c)[0]
                if len(class_indices) < k: continue

                replace = len(class_indices) < k
                train_indices_c = np.random.choice(class_indices, k, replace=replace)
                train_indices_c = np.unique(train_indices_c)
                while len(train_indices_c) < k and replace:
                     additional_sample = np.random.choice(class_indices, 1)[0]
                     if additional_sample not in train_indices_c:
                         train_indices_c = np.append(train_indices_c, additional_sample)
                if len(train_indices_c) > k: train_indices_c = train_indices_c[:k]
                if len(train_indices_c) < k: continue

                test_indices_c = np.setdiff1d(class_indices, train_indices_c)

                if len(test_indices_c) > 0:
                    X_train_fs.append(embeddings_scaled[train_indices_c])
                    y_train_fs.append(labels[train_indices_c])
                    X_test_fs.append(embeddings_scaled[test_indices_c])
                    y_test_fs.append(labels[test_indices_c])

            if not X_train_fs or len(X_train_fs) < n_classes: continue

            X_train = np.concatenate(X_train_fs)
            y_train = np.concatenate(y_train_fs)
            X_test = np.concatenate(X_test_fs)
            y_test = np.concatenate(y_test_fs)

            if len(X_test) == 0: continue

            n_neighbors_fs = min(k, len(X_train))
            n_neighbors_fs = max(1, n_neighbors_fs)
            classifier = KNeighborsClassifier(n_neighbors=n_neighbors_fs)
            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)
            accuracies.append(accuracy_score(y_test, y_pred))

        if accuracies:
            mean_acc = np.mean(accuracies)
            results[f'{k}_shot_accuracy'] = mean_acc
            logging.info(f"{k}-shot Average Accuracy ({len(accuracies)} valid runs): {mean_acc:.4f}")
        else:
            results[f'{k}_shot_accuracy'] = 0.0
            logging.warning(f"{k}-shot evaluation skipped or failed due to insufficient data across runs.")

    return results


def evaluate_ssl_model(encoder, config, class_names, dataset):
    """Complete evaluation of a self-supervised model using multiple metrics"""
    eval_output_dir = os.path.join(config['output_dir'], 'evaluation_results')
    os.makedirs(eval_output_dir, exist_ok=True)
    logging.info("Starting SSL Model Evaluation")
    logging.info(f"Results will be saved in: {eval_output_dir}")

    eval_dataloader, _, _, _ = get_dataloader(config, shuffle=False)
    if eval_dataloader is None:
        logging.error("Failed evaluation dataloader.")
        return None

    embeddings, labels = generate_embeddings(encoder, eval_dataloader, config)
    if embeddings is None:
        logging.error("Failed embeddings generation.")
        return None

    if len(np.unique(labels)) != len(class_names):
        logging.warning(f"Warning: Label/Class mismatch.")

    results = {}
    n_classes = len(class_names)

    try:
        results['linear_eval_accuracy'] = linear_evaluation(embeddings, labels, config)
    except Exception as e:
        logging.error(f"Error Linear Eval: {e}")
        results['linear_eval_accuracy'] = 0.0

    try:
        results['nearest_neighbor_accuracy'] = nearest_neighbor_evaluation(embeddings, labels, config)
    except Exception as e:
        logging.error(f"Error k-NN Eval: {e}")
        results['nearest_neighbor_accuracy'] = 0.0

    try:
        results.update(evaluate_clustering(embeddings, labels, config, n_classes=n_classes))
    except Exception as e:
        logging.error(f"Error Clustering: {e}")
        results.update({'nmi': 0.0, 'ari': 0.0})

    try:
        results.update(few_shot_evaluation(embeddings, labels, config, n_classes=n_classes))
    except Exception as e:
        logging.error(f"Error Few-Shot: {e}")
        [results.update({f'{k}_shot_accuracy': 0.0}) for k in config['eval_few_shot_shots']]

    # Create summary visualizations
    create_evaluation_summary(results, eval_output_dir, config)

    logging.info("--- Evaluation Summary ---")
    for metric, value in results.items():
        logging.info(f"{metric}: {value:.4f}")

    return results