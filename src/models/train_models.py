#!/usr/bin/env python3
"""
Training script for Network Traffic Anomaly Detection Models

This script trains all anomaly detection models and saves them for use in the real-time pipeline.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from src.data_collection.collector import NetworkDataCollector
from src.data_processing.processor import NetworkDataProcessor
from src.models.anomaly_detector import AnomalyDetector
from config.settings import MODEL_CONFIG, DATA_COLLECTION_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main training function"""
    print("🤖 Training Network Traffic Anomaly Detection Models")
    print("=" * 60)
    
    # Initialize components
    collector = NetworkDataCollector()
    processor = NetworkDataProcessor()
    detector = AnomalyDetector()
    
    # Generate or load training data
    logger.info("Preparing training data...")
    
    # Try to load CICIDS2017 data first
    training_data = None
    for day in DATA_COLLECTION_CONFIG['cicids2017']['days']:
        data = collector.load_cicids2017_data(day)
        if data is not None:
            training_data = data
            logger.info(f"Loaded CICIDS2017 {day} data: {len(data)} samples")
            break
    
    # If no CICIDS2017 data available, generate synthetic data
    if training_data is None:
        logger.info("CICIDS2017 data not available, generating synthetic data...")
        training_data = collector.generate_synthetic_data(
            DATA_COLLECTION_CONFIG['synthetic_data']['n_samples']
        )
        logger.info(f"Generated synthetic data: {len(training_data)} samples")
    
    if training_data.empty:
        logger.error("No training data available")
        return False
    
    # Process the data
    logger.info("Processing training data...")
    processed_data = processor.parse_cicids2017_format(training_data)
    
    # Prepare features for ML
    X, y = processor.prepare_features_for_ml(processed_data)
    
    if X.empty:
        logger.error("Failed to prepare features for training")
        return False
    
    logger.info(f"Prepared {X.shape[1]} features for training")
    
    # Train individual models
    logger.info("Training individual models...")
    
    # Train Isolation Forest
    logger.info("Training Isolation Forest...")
    isolation_results = detector.train_isolation_forest(
        X, 
        contamination=MODEL_CONFIG['isolation_forest']['contamination']
    )
    
    # Train Autoencoder
    logger.info("Training Autoencoder...")
    autoencoder_results = detector.train_autoencoder(
        X,
        encoding_dim=MODEL_CONFIG['autoencoder']['encoding_dim'],
        epochs=MODEL_CONFIG['autoencoder']['epochs'],
        batch_size=MODEL_CONFIG['autoencoder']['batch_size']
    )
    
    # Train K-Means
    logger.info("Training K-Means...")
    kmeans_results = detector.train_kmeans_clustering(
        X,
        n_clusters=MODEL_CONFIG['kmeans']['n_clusters']
    )
    
    # Train DBSCAN
    logger.info("Training DBSCAN...")
    dbscan_results = detector.train_dbscan_clustering(
        X,
        eps=MODEL_CONFIG['dbscan']['eps'],
        min_samples=MODEL_CONFIG['dbscan']['min_samples']
    )
    
    # Train Ensemble Model
    logger.info("Training Ensemble Model...")
    ensemble_results = detector.train_ensemble_model(X, y)
    
    # Evaluate models if we have ground truth
    if y is not None:
        logger.info("Evaluating models...")
        
        models_to_evaluate = ['isolation_forest', 'autoencoder', 'kmeans', 'ensemble']
        
        for model_name in models_to_evaluate:
            if model_name in detector.models:
                evaluation = detector.evaluate_model(X, y, model_name)
                if evaluation:
                    logger.info(f"{model_name.upper()}:")
                    logger.info(f"  Accuracy: {evaluation['accuracy']:.3f}")
                    logger.info(f"  Precision: {evaluation['precision']:.3f}")
                    logger.info(f"  Recall: {evaluation['recall']:.3f}")
                    logger.info(f"  F1-Score: {evaluation['f1_score']:.3f}")
    
    # Save trained models
    logger.info("Saving trained models...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = f"anomaly_detector_{timestamp}"
    
    if detector.save_models(filename_prefix):
        logger.info("Models saved successfully")
        
        # Save training metadata
        metadata = {
            'training_timestamp': timestamp,
            'training_samples': len(X),
            'features_count': X.shape[1],
            'models_trained': list(detector.models.keys()),
            'data_source': 'CICIDS2017' if 'cicids2017' in str(training_data) else 'Synthetic',
            'model_config': MODEL_CONFIG
        }
        
        metadata_path = detector.models_dir / f"{filename_prefix}_metadata.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Training metadata saved to {metadata_path}")
        
        return True
    else:
        logger.error("Failed to save models")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Model training completed successfully!")
        print("\nNext steps:")
        print("1. Start the dashboard: streamlit run src/dashboard/app.py")
        print("2. Start the streaming pipeline: python src/streaming/pipeline.py")
    else:
        print("\n❌ Model training failed!")
        sys.exit(1)
