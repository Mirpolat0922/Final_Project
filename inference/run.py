"""
Inference Script
Loads trained model and generates predictions on inference data.
"""

import argparse
import json
import logging
import os
import pickle
import sys
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

CONF_FILE = os.getenv('CONF_PATH', 'settings.json')

from utils import get_project_dir, configure_logging, download_nltk_resources, preprocess_text

# Load configuration
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
RESULTS_DIR = get_project_dir(conf['general']['results_dir'])

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--infer_file",
                    help="Specify inference data file",
                    default=conf['inference']['table_name'])
parser.add_argument("--out_path",
                    help="Specify the path to the output table")


def load_model_artifacts():
    """Load trained model and vectorizer."""
    logging.info("Loading model artifacts...")

    model_path = os.path.join(MODEL_DIR, conf['inference']['model_name'])
    vectorizer_path = os.path.join(MODEL_DIR, conf['inference']['vectorizer_name'])

    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        logging.info(f"Model loaded from {model_path}")

        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        logging.info(f"Vectorizer loaded from {vectorizer_path}")

        return model, vectorizer
    except Exception as e:
        logging.error(f"Error loading model artifacts: {e}")
        sys.exit(1)


def get_inference_data(path: str) -> pd.DataFrame:
    """Load inference data from CSV."""
    try:
        logging.info(f"Loading inference data from {path}...")
        df = pd.read_csv(path)
        logging.info(f"Loaded {len(df)} samples")
        return df
    except Exception as e:
        logging.error(f"Error loading inference data: {e}")
        sys.exit(1)


def preprocess_inference_data(df: pd.DataFrame) -> tuple:
    """Preprocess inference data."""
    logging.info("Preprocessing inference data...")

    df = df.copy()
    keep_negations = conf['text_processing']['keep_negations']

    # Preprocess text
    df['processed_text'] = df['review'].apply(
        lambda x: preprocess_text(x, keep_negations)
    )

    # Convert sentiment if exists
    y = None
    if 'sentiment' in df.columns:
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        y = df['sentiment']

    logging.info("Preprocessing complete!")
    return df['processed_text'], y, df


def predict_results(model, vectorizer, X):
    """Generate predictions."""
    logging.info("Generating predictions...")

    # Transform to TF-IDF
    X_tfidf = vectorizer.transform(X)

    # Predict
    predictions = model.predict(X_tfidf)
    decision_scores = model.decision_function(X_tfidf)

    logging.info(f"Generated {len(predictions)} predictions")
    return predictions, decision_scores


def evaluate_predictions(y_true, y_pred, y_proba):
    """Evaluate predictions if ground truth available."""
    logging.info("\nEvaluating predictions...")

    accuracy = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_proba)

    logging.info(f"Inference Accuracy: {accuracy:.5f}")
    logging.info(f"Inference ROC AUC: {roc_auc:.5f}")

    logging.info("\nClassification Report:")
    logging.info("\n" + classification_report(y_true, y_pred,
                                             target_names=['negative', 'positive']))

    # Save confusion matrix
    save_confusion_matrix(y_true, y_pred)

    # Save metrics
    save_metrics(accuracy, roc_auc, y_true, y_pred)


def save_confusion_matrix(y_true, y_pred):
    """Save confusion matrix plot."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Inference Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    figures_dir = os.path.join(RESULTS_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    cm_path = os.path.join(figures_dir, 'inference_confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    logging.info(f"Confusion matrix saved to {cm_path}")
    plt.close()


def save_metrics(accuracy, roc_auc, y_true, y_pred):
    """Save metrics to file."""
    metrics = {
        'accuracy': float(accuracy),
        'roc_auc': float(roc_auc),
        'classification_report': classification_report(
            y_true, y_pred,
            target_names=['negative', 'positive'],
            output_dict=True
        )
    }

    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    metrics_path = os.path.join(RESULTS_DIR, 'inference_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics saved to {metrics_path}")


def store_results(predictions, decision_scores, original_df, path=None):
    """Store prediction results."""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Create results dataframe
    results_df = pd.DataFrame({
        'review': original_df['review'].values,
        'true_sentiment': original_df['sentiment'].map({1: 'positive', 0: 'negative'}).values if 'sentiment' in original_df.columns else None,
        'predicted_sentiment': ['positive' if p == 1 else 'negative' for p in predictions],
        'decision_score': decision_scores
    })

    if not path:
        path = datetime.now().strftime(conf['general']['datetime_format']) + '.csv'
        path = os.path.join(RESULTS_DIR, path)

    results_df.to_csv(path, index=False)
    logging.info(f'Results saved to {path}')


def main():
    """Main inference pipeline."""
    configure_logging()
    args = parser.parse_args()

    logging.info("="*60)
    logging.info("SENTIMENT ANALYSIS MODEL INFERENCE")
    logging.info("="*60)

    # Download NLTK resources
    logging.info("\nDownloading NLTK resources...")
    download_nltk_resources()

    # Load model
    logging.info("\n" + "="*60)
    logging.info("Loading Model Artifacts")
    logging.info("="*60)
    model, vectorizer = load_model_artifacts()

    # Load data
    logging.info("\n" + "="*60)
    logging.info("Loading Inference Data")
    logging.info("="*60)
    infer_file = args.infer_file
    infer_data = get_inference_data(os.path.join(DATA_DIR, 'raw', infer_file))

    # Preprocess
    logging.info("\n" + "="*60)
    logging.info("Preprocessing Data")
    logging.info("="*60)
    X, y, original_df = preprocess_inference_data(infer_data)

    # Predict
    logging.info("\n" + "="*60)
    logging.info("Generating Predictions")
    logging.info("="*60)
    predictions, decision_scores = predict_results(model, vectorizer, X)

    # Evaluate if ground truth available
    if y is not None:
        logging.info("\n" + "="*60)
        logging.info("Evaluating Predictions")
        logging.info("="*60)
        evaluate_predictions(y, predictions, decision_scores)

    # Store results
    logging.info("\n" + "="*60)
    logging.info("Saving Results")
    logging.info("="*60)
    store_results(predictions, decision_scores, original_df, args.out_path)

    logging.info("\n" + "="*60)
    logging.info("INFERENCE COMPLETE!")
    logging.info("="*60)


if __name__ == "__main__":
    main()