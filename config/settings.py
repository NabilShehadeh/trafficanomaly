"""
Configuration settings for the Network Traffic Anomaly Detection System
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Kafka Configuration
KAFKA_CONFIG = {
    'bootstrap_servers': os.getenv('KAFKA_BOOTSTRAP_SERVERS', 'localhost:9092'),
    'topic': os.getenv('KAFKA_TOPIC', 'network-traffic'),
    'group_id': os.getenv('KAFKA_GROUP_ID', 'anomaly-detection-group'),
    'auto_offset_reset': 'latest',
    'enable_auto_commit': True,
    'session_timeout_ms': 30000,
    'heartbeat_interval_ms': 10000
}

# Spark Configuration
SPARK_CONFIG = {
    'app_name': 'NetworkTrafficAnomalyDetection',
    'master': os.getenv('SPARK_MASTER', 'local[*]'),
    'config': {
        'spark.sql.adaptive.enabled': 'true',
        'spark.sql.adaptive.coalescePartitions.enabled': 'true',
        'spark.streaming.kafka.maxRatePerPartition': '1000',
        'spark.streaming.backpressure.enabled': 'true'
    }
}

# Model Configuration
MODEL_CONFIG = {
    'isolation_forest': {
        'contamination': 0.1,
        'n_estimators': 100,
        'max_samples': 'auto',
        'max_features': 1.0,
        'random_state': 42
    },
    'autoencoder': {
        'encoding_dim': 32,
        'epochs': 100,
        'batch_size': 32,
        'learning_rate': 0.001,
        'validation_split': 0.2
    },
    'kmeans': {
        'n_clusters': 8,
        'random_state': 42,
        'n_init': 10
    },
    'dbscan': {
        'eps': 0.5,
        'min_samples': 5
    }
}

# Data Collection Configuration
DATA_COLLECTION_CONFIG = {
    'cicids2017': {
        'dataset_url': 'https://www.unb.ca/cic/datasets/ids-2017.html',
        'data_dir': DATA_DIR / 'cicids2017',
        'days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    },
    'packet_capture': {
        'interface': None,  # None for default interface
        'packet_count': 1000,
        'capture_dir': DATA_DIR / 'captured',
        'file_format': 'csv'
    },
    'synthetic_data': {
        'n_samples': 10000,
        'anomaly_rate': 0.05,
        'random_state': 42
    }
}

# Dashboard Configuration
DASHBOARD_CONFIG = {
    'title': 'Network Traffic Anomaly Detection',
    'page_config': {
        'page_title': 'Network Traffic Anomaly Detection',
        'page_icon': '🛡️',
        'layout': 'wide',
        'initial_sidebar_state': 'expanded'
    },
    'refresh_interval': 5,  # seconds
    'max_alerts_display': 100
}

# Alert Configuration
ALERT_CONFIG = {
    'severity_levels': ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
    'alert_types': [
        'Network Anomaly',
        'DDoS Attack',
        'Port Scan',
        'Suspicious Traffic',
        'Protocol Anomaly'
    ],
    'notification_channels': ['dashboard', 'email', 'webhook'],
    'retention_days': 30
}

# Logging Configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOGS_DIR / 'anomaly_detection.log',
    'max_bytes': 10 * 1024 * 1024,  # 10MB
    'backup_count': 5
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    'batch_size': 1000,
    'processing_interval': 10,  # seconds
    'max_memory_usage': '2GB',
    'parallel_processing': True,
    'num_workers': 4
}

# Security Configuration
SECURITY_CONFIG = {
    'allowed_interfaces': ['eth0', 'wlan0', 'lo'],
    'max_packet_size': 65535,
    'rate_limit_per_ip': 1000,  # packets per minute
    'blacklisted_ips': [],
    'whitelisted_ips': ['127.0.0.1', '::1']
}

# Environment-specific configurations
ENVIRONMENTS = {
    'development': {
        'debug': True,
        'log_level': 'DEBUG',
        'kafka_bootstrap_servers': 'localhost:9092',
        'spark_master': 'local[*]'
    },
    'production': {
        'debug': False,
        'log_level': 'INFO',
        'kafka_bootstrap_servers': 'kafka-cluster:9092',
        'spark_master': 'spark://spark-master:7077'
    },
    'testing': {
        'debug': True,
        'log_level': 'WARNING',
        'kafka_bootstrap_servers': 'localhost:9092',
        'spark_master': 'local[1]'
    }
}

# Get current environment
CURRENT_ENV = os.getenv('ENVIRONMENT', 'development')
ENV_CONFIG = ENVIRONMENTS.get(CURRENT_ENV, ENVIRONMENTS['development'])

# Export all configurations
__all__ = [
    'BASE_DIR', 'DATA_DIR', 'MODELS_DIR', 'LOGS_DIR',
    'KAFKA_CONFIG', 'SPARK_CONFIG', 'MODEL_CONFIG',
    'DATA_COLLECTION_CONFIG', 'DASHBOARD_CONFIG',
    'ALERT_CONFIG', 'LOGGING_CONFIG', 'PERFORMANCE_CONFIG',
    'SECURITY_CONFIG', 'ENV_CONFIG', 'CURRENT_ENV'
]
