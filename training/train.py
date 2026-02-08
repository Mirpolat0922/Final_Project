"""
Training Script
Prepares data, trains sentiment analysis model, and saves artifacts.
"""

import argparse
import os
import sys
import pickle
import json
import logging
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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
TRAIN_PATH = os.path.join(DATA_DIR, 'raw', conf['train']['table_name'])

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("--train_file",
                    help="Specify training data file",
                    default=conf['train']['table_name'])


class DataProcessor:
    """Handles data loading and preprocessing."""

    def __init__(self):
        self.keep_negations = conf['text_processing']['keep_negations']

    def prepare_data(self) -> tuple:
        """Load and preprocess training data."""
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        X, y = self.preprocess_data(df)
        return X, y

    def data_extraction(self, path: str) -> pd.DataFrame:
        """Load data from CSV."""
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)

    def preprocess_data(self, df: pd.DataFrame) -> tuple:
        """Preprocess text and convert labels."""
        logging.info("Preprocessing text data...")

        # Convert sentiment to binary
        df = df.copy()
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

        # Preprocess text
        logging.info(f"Preprocessing {len(df)} reviews...")
        df['processed_text'] = df['review'].apply(
            lambda x: preprocess_text(x, self.keep_negations)
        )

        logging.info("Preprocessing complete!")
        return df['processed_text'], df['sentiment']


class Training:
    """Handles model training and evaluation."""

    def __init__(self):
        self.model = LinearSVC(
            max_iter=conf['train']['svm_max_iter'],
            random_state=conf['general']['random_state']
        )
        self.vectorizer = TfidfVectorizer(
            ngram_range=tuple(conf['train']['tfidf_ngram_range']),
            max_df=conf['train']['tfidf_max_df'],
            min_df=conf['train']['tfidf_min_df']
        )

    def run_training(self, X: pd.Series, y: pd.Series) -> None:
        """Execute the training pipeline."""
        logging.info("="*60)
        logging.info("Creating TF-IDF Features")
        logging.info("="*60)
        X_tfidf = self.create_features(X)

        logging.info("\n" + "="*60)
        logging.info("Training Model")
        logging.info("="*60)
        self.train(X_tfidf, y)

        logging.info("\n" + "="*60)
        logging.info("Evaluating Model")
        logging.info("="*60)
        self.evaluate(X_tfidf, y)

        logging.info("\n" + "="*60)
        logging.info("Saving Model Artifacts")
        logging.info("="*60)
        self.save()

    def create_features(self, X: pd.Series):
        """Create TF-IDF features."""
        logging.info("Vectorizing text with TF-IDF...")
        X_tfidf = self.vectorizer.fit_transform(X)
        logging.info(f"TF-IDF shape: {X_tfidf.shape}")
        return X_tfidf

    def train(self, X_train, y_train) -> None:
        """Train the model."""
        logging.info("Training LinearSVC model...")
        start_time = time.time()
        self.model.fit(X_train, y_train)
        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds")

    def evaluate(self, X, y) -> None:
        """Evaluate the model."""
        logging.info("Evaluating model on training data...")
        y_pred = self.model.predict(X)
        y_proba = self.model.decision_function(X)

        accuracy = accuracy_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_proba)

        logging.info(f"Training Accuracy: {accuracy:.5f}")
        logging.info(f"Training ROC AUC: {roc_auc:.5f}")

        # Save confusion matrix
        self.save_confusion_matrix(y, y_pred)

        # Save metrics
        self.save_metrics(accuracy, roc_auc, y, y_pred)

    def save_confusion_matrix(self, y_true, y_pred):
        """Save confusion matrix plot."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'])
        plt.title('Training Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()

        figures_dir = os.path.join(MODEL_DIR, 'figures')
        os.makedirs(figures_dir, exist_ok=True)

        cm_path = os.path.join(figures_dir, 'confusion_matrix.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        logging.info(f"Confusion matrix saved to {cm_path}")
        plt.close()

    def save_metrics(self, accuracy, roc_auc, y_true, y_pred):
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

        metrics_path = os.path.join(MODEL_DIR, 'training_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        logging.info(f"Metrics saved to {metrics_path}")

    def save(self) -> None:
        """Save model and vectorizer."""
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        # Save model
        model_path = os.path.join(MODEL_DIR, conf['inference']['model_name'])
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        logging.info(f"Model saved to {model_path}")

        # Save vectorizer
        vectorizer_path = os.path.join(MODEL_DIR, conf['inference']['vectorizer_name'])
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        logging.info(f"Vectorizer saved to {vectorizer_path}")


def main():
    """Main training pipeline."""
    configure_logging()

    logging.info("="*60)
    logging.info("SENTIMENT ANALYSIS MODEL TRAINING")
    logging.info("="*60)

    # Download NLTK resources
    logging.info("\nDownloading NLTK resources...")
    download_nltk_resources()

    # Initialize processors
    data_proc = DataProcessor()
    trainer = Training()

    # Load and preprocess data
    X, y = data_proc.prepare_data()

    # Train model
    trainer.run_training(X, y)

    logging.info("\n" + "="*60)
    logging.info("TRAINING COMPLETE!")
    logging.info("="*60)


if __name__ == "__main__":
    main()