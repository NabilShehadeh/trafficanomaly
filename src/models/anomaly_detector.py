"""
Predictive Models Module for Network Traffic Anomaly Detection

This module implements multiple anomaly detection algorithms:
1. Isolation Forest
2. Autoencoder (Deep Learning)
3. Clustering-based methods (K-Means, DBSCAN)
4. Ensemble methods
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Any, Optional, Tuple
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Deep Learning Libraries
try:
    import tensorflow as tf
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import Dense, Input, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

# Additional ML Libraries
try:
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    PYOD_AVAILABLE = True
except ImportError:
    PYOD_AVAILABLE = False
    print("Warning: PyOD not available. Install with: pip install pyod")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AnomalyDetector:
    """Main class for anomaly detection using multiple algorithms"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.models = {}
        self.scalers = {}
        self.model_scores = {}
        
    def train_isolation_forest(self, X: pd.DataFrame, contamination: float = 0.1) -> Dict[str, Any]:
        """
        Train Isolation Forest model
        
        Args:
            X: Feature matrix
            contamination: Expected proportion of anomalies
            
        Returns:
            Dict: Model results and metrics
        """
        try:
            logger.info("Training Isolation Forest model...")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train model
            model = IsolationForest(
                contamination=contamination,
                random_state=42,
                n_estimators=100,
                max_samples='auto',
                max_features=1.0
            )
            
            model.fit(X_scaled)
            
            # Predict anomalies
            predictions = model.predict(X_scaled)
            anomaly_scores = model.decision_function(X_scaled)
            
            # Convert predictions to binary (1 = anomaly, 0 = normal)
            binary_predictions = (predictions == -1).astype(int)
            
            # Store model and scaler
            self.models['isolation_forest'] = model
            self.scalers['isolation_forest'] = scaler
            
            # Calculate metrics
            results = {
                'model': 'Isolation Forest',
                'anomaly_count': np.sum(binary_predictions),
                'anomaly_rate': np.mean(binary_predictions),
                'mean_anomaly_score': np.mean(anomaly_scores),
                'predictions': binary_predictions,
                'anomaly_scores': anomaly_scores
            }
            
            logger.info(f"Isolation Forest trained. Detected {results['anomaly_count']} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"Error training Isolation Forest: {e}")
            return {}
    
    def train_autoencoder(self, X: pd.DataFrame, encoding_dim: int = 32, 
                         epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train Autoencoder model for anomaly detection
        
        Args:
            X: Feature matrix
            encoding_dim: Dimension of encoded representation
            epochs: Number of training epochs
            batch_size: Batch size for training
            
        Returns:
            Dict: Model results and metrics
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available for Autoencoder training")
            return {}
        
        try:
            logger.info("Training Autoencoder model...")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            input_dim = X_scaled.shape[1]
            
            # Build autoencoder
            input_layer = Input(shape=(input_dim,))
            encoder = Dense(encoding_dim * 2, activation="relu")(input_layer)
            encoder = Dropout(0.2)(encoder)
            encoder = Dense(encoding_dim, activation="relu")(encoder)
            
            decoder = Dense(encoding_dim * 2, activation="relu")(encoder)
            decoder = Dropout(0.2)(decoder)
            decoder = Dense(input_dim, activation="sigmoid")(decoder)
            
            autoencoder = Model(inputs=input_layer, outputs=decoder)
            autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            
            # Train autoencoder
            history = autoencoder.fit(
                X_scaled, X_scaled,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5)
                ],
                verbose=0
            )
            
            # Calculate reconstruction error
            reconstructed = autoencoder.predict(X_scaled)
            reconstruction_error = np.mean(np.square(X_scaled - reconstructed), axis=1)
            
            # Determine threshold (95th percentile)
            threshold = np.percentile(reconstruction_error, 95)
            binary_predictions = (reconstruction_error > threshold).astype(int)
            
            # Store model and scaler
            self.models['autoencoder'] = autoencoder
            self.scalers['autoencoder'] = scaler
            
            # Calculate metrics
            results = {
                'model': 'Autoencoder',
                'anomaly_count': np.sum(binary_predictions),
                'anomaly_rate': np.mean(binary_predictions),
                'mean_reconstruction_error': np.mean(reconstruction_error),
                'threshold': threshold,
                'predictions': binary_predictions,
                'reconstruction_errors': reconstruction_error,
                'training_history': history.history
            }
            
            logger.info(f"Autoencoder trained. Detected {results['anomaly_count']} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"Error training Autoencoder: {e}")
            return {}
    
    def train_kmeans_clustering(self, X: pd.DataFrame, n_clusters: int = 8) -> Dict[str, Any]:
        """
        Train K-Means clustering for anomaly detection
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters
            
        Returns:
            Dict: Model results and metrics
        """
        try:
            logger.info("Training K-Means clustering model...")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train K-Means
            model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            model.fit(X_scaled)
            
            # Calculate distances to centroids
            distances = model.transform(X_scaled)
            min_distances = np.min(distances, axis=1)
            
            # Determine threshold (95th percentile)
            threshold = np.percentile(min_distances, 95)
            binary_predictions = (min_distances > threshold).astype(int)
            
            # Store model and scaler
            self.models['kmeans'] = model
            self.scalers['kmeans'] = scaler
            
            # Calculate metrics
            results = {
                'model': 'K-Means Clustering',
                'anomaly_count': np.sum(binary_predictions),
                'anomaly_rate': np.mean(binary_predictions),
                'mean_distance': np.mean(min_distances),
                'threshold': threshold,
                'predictions': binary_predictions,
                'distances': min_distances,
                'cluster_labels': model.labels_
            }
            
            logger.info(f"K-Means trained. Detected {results['anomaly_count']} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"Error training K-Means: {e}")
            return {}
    
    def train_dbscan_clustering(self, X: pd.DataFrame, eps: float = 0.5, 
                              min_samples: int = 5) -> Dict[str, Any]:
        """
        Train DBSCAN clustering for anomaly detection
        
        Args:
            X: Feature matrix
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
            
        Returns:
            Dict: Model results and metrics
        """
        try:
            logger.info("Training DBSCAN clustering model...")
            
            # Scale features
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Train DBSCAN
            model = DBSCAN(eps=eps, min_samples=min_samples)
            cluster_labels = model.fit_predict(X_scaled)
            
            # Anomalies are points labeled as -1
            binary_predictions = (cluster_labels == -1).astype(int)
            
            # Store model and scaler
            self.models['dbscan'] = model
            self.scalers['dbscan'] = scaler
            
            # Calculate metrics
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = list(cluster_labels).count(-1)
            
            results = {
                'model': 'DBSCAN Clustering',
                'anomaly_count': n_noise,
                'anomaly_rate': n_noise / len(X),
                'n_clusters': n_clusters,
                'predictions': binary_predictions,
                'cluster_labels': cluster_labels
            }
            
            logger.info(f"DBSCAN trained. Detected {results['anomaly_count']} anomalies")
            return results
            
        except Exception as e:
            logger.error(f"Error training DBSCAN: {e}")
            return {}
    
    def train_ensemble_model(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Train ensemble of multiple anomaly detection models
        
        Args:
            X: Feature matrix
            y: Target labels (optional)
            
        Returns:
            Dict: Ensemble results and metrics
        """
        try:
            logger.info("Training ensemble model...")
            
            # Train individual models
            isolation_results = self.train_isolation_forest(X)
            autoencoder_results = self.train_autoencoder(X)
            kmeans_results = self.train_kmeans_clustering(X)
            
            # Combine predictions (majority voting)
            predictions_list = [
                isolation_results.get('predictions', []),
                autoencoder_results.get('predictions', []),
                kmeans_results.get('predictions', [])
            ]
            
            # Remove empty predictions
            predictions_list = [pred for pred in predictions_list if len(pred) > 0]
            
            if predictions_list:
                # Majority voting
                ensemble_predictions = np.mean(predictions_list, axis=0) > 0.5
                ensemble_predictions = ensemble_predictions.astype(int)
                
                # Calculate ensemble metrics
                results = {
                    'model': 'Ensemble',
                    'anomaly_count': np.sum(ensemble_predictions),
                    'anomaly_rate': np.mean(ensemble_predictions),
                    'predictions': ensemble_predictions,
                    'individual_results': {
                        'isolation_forest': isolation_results,
                        'autoencoder': autoencoder_results,
                        'kmeans': kmeans_results
                    }
                }
                
                logger.info(f"Ensemble trained. Detected {results['anomaly_count']} anomalies")
                return results
            else:
                logger.error("No individual models trained successfully")
                return {}
                
        except Exception as e:
            logger.error(f"Error training ensemble model: {e}")
            return {}
    
    def predict_anomalies(self, X: pd.DataFrame, model_name: str = 'ensemble') -> Dict[str, Any]:
        """
        Predict anomalies using trained model
        
        Args:
            X: Feature matrix
            model_name: Name of model to use
            
        Returns:
            Dict: Prediction results
        """
        try:
            if model_name not in self.models:
                logger.error(f"Model {model_name} not found")
                return {}
            
            model = self.models[model_name]
            scaler = self.scalers.get(model_name)
            
            if scaler:
                X_scaled = scaler.transform(X)
            else:
                X_scaled = X.values
            
            if model_name == 'isolation_forest':
                predictions = model.predict(X_scaled)
                anomaly_scores = model.decision_function(X_scaled)
                binary_predictions = (predictions == -1).astype(int)
                
                return {
                    'predictions': binary_predictions,
                    'anomaly_scores': anomaly_scores,
                    'model': model_name
                }
            
            elif model_name == 'autoencoder':
                reconstructed = model.predict(X_scaled)
                reconstruction_error = np.mean(np.square(X_scaled - reconstructed), axis=1)
                threshold = np.percentile(reconstruction_error, 95)
                binary_predictions = (reconstruction_error > threshold).astype(int)
                
                return {
                    'predictions': binary_predictions,
                    'reconstruction_errors': reconstruction_error,
                    'threshold': threshold,
                    'model': model_name
                }
            
            elif model_name == 'kmeans':
                distances = model.transform(X_scaled)
                min_distances = np.min(distances, axis=1)
                threshold = np.percentile(min_distances, 95)
                binary_predictions = (min_distances > threshold).astype(int)
                
                return {
                    'predictions': binary_predictions,
                    'distances': min_distances,
                    'threshold': threshold,
                    'model': model_name
                }
            
            else:
                logger.error(f"Prediction not implemented for {model_name}")
                return {}
                
        except Exception as e:
            logger.error(f"Error predicting anomalies: {e}")
            return {}
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series, model_name: str = 'ensemble') -> Dict[str, Any]:
        """
        Evaluate model performance
        
        Args:
            X: Feature matrix
            y: True labels
            model_name: Name of model to evaluate
            
        Returns:
            Dict: Evaluation metrics
        """
        try:
            predictions = self.predict_anomalies(X, model_name)
            
            if not predictions or 'predictions' not in predictions:
                logger.error(f"No predictions available for {model_name}")
                return {}
            
            pred = predictions['predictions']
            
            # Calculate metrics
            accuracy = np.mean(pred == y)
            precision = np.sum((pred == 1) & (y == 1)) / (np.sum(pred == 1) + 1e-8)
            recall = np.sum((pred == 1) & (y == 1)) / (np.sum(y == 1) + 1e-8)
            f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            
            evaluation = {
                'model': model_name,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'confusion_matrix': {
                    'true_negatives': tn,
                    'false_positives': fp,
                    'false_negatives': fn,
                    'true_positives': tp
                }
            }
            
            logger.info(f"Model {model_name} evaluation: F1={f1_score:.3f}, Accuracy={accuracy:.3f}")
            return evaluation
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def save_models(self, filename_prefix: str = "anomaly_detector") -> bool:
        """Save trained models to disk"""
        try:
            for model_name, model in self.models.items():
                model_path = self.models_dir / f"{filename_prefix}_{model_name}.joblib"
                joblib.dump(model, model_path)
                logger.info(f"Saved {model_name} model to {model_path}")
            
            # Save scalers
            for scaler_name, scaler in self.scalers.items():
                scaler_path = self.models_dir / f"{filename_prefix}_{scaler_name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
                logger.info(f"Saved {scaler_name} scaler to {scaler_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return False
    
    def load_models(self, filename_prefix: str = "anomaly_detector") -> bool:
        """Load trained models from disk"""
        try:
            model_files = list(self.models_dir.glob(f"{filename_prefix}_*.joblib"))
            
            for model_file in model_files:
                if "_scaler" not in str(model_file):
                    model_name = model_file.stem.replace(f"{filename_prefix}_", "")
                    model = joblib.load(model_file)
                    self.models[model_name] = model
                    logger.info(f"Loaded {model_name} model from {model_file}")
                else:
                    scaler_name = model_file.stem.replace(f"{filename_prefix}_", "").replace("_scaler", "")
                    scaler = joblib.load(model_file)
                    self.scalers[scaler_name] = scaler
                    logger.info(f"Loaded {scaler_name} scaler from {model_file}")
            
            return len(self.models) > 0
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False

def main():
    """Example usage of AnomalyDetector"""
    detector = AnomalyDetector()
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # Normal data
    X_normal = np.random.normal(0, 1, (n_samples, n_features))
    
    # Anomalous data
    X_anomaly = np.random.normal(5, 2, (50, n_features))
    
    # Combine data
    X = np.vstack([X_normal, X_anomaly])
    y = np.hstack([np.zeros(n_samples), np.ones(50)])
    
    # Convert to DataFrame
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    
    print("Training anomaly detection models...")
    
    # Train different models
    isolation_results = detector.train_isolation_forest(df)
    autoencoder_results = detector.train_autoencoder(df)
    kmeans_results = detector.train_kmeans_clustering(df)
    ensemble_results = detector.train_ensemble_model(df)
    
    # Evaluate models
    print("\nModel Evaluation:")
    for model_name in ['isolation_forest', 'autoencoder', 'kmeans', 'ensemble']:
        if model_name in detector.models:
            evaluation = detector.evaluate_model(df, pd.Series(y), model_name)
            if evaluation:
                print(f"{model_name}: F1={evaluation['f1_score']:.3f}, "
                      f"Accuracy={evaluation['accuracy']:.3f}")

if __name__ == "__main__":
    main()
