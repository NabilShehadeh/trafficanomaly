import sys
import argparse
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from src.data_collection.collector import NetworkDataCollector
from src.data_processing.processor import NetworkDataProcessor
from src.models.anomaly_detector import AnomalyDetector
from src.streaming.pipeline import RealTimePipeline
from src.models.train_models import main as train_models

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_dashboard():
    try:
        import subprocess

        dashboard_path = Path(__file__).parent / "src" / "dashboard" / "app.py"

        if not dashboard_path.exists():
            logger.error(f"Dashboard not found at {dashboard_path}")
            return False

        logger.info("Starting Streamlit dashboard at http://localhost:8501")

        subprocess.run([
            sys.executable, "-m", "streamlit", "run",
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=0.0.0.0"
        ])

        return True

    except ImportError:
        logger.error("Streamlit not installed. Install with: pip install streamlit")
        return False
    except Exception as e:
        logger.error(f"Error starting dashboard: {e}")
        return False


def run_pipeline():
    try:
        logger.info("Starting real-time anomaly detection pipeline...")

        pipeline = RealTimePipeline()
        pipeline.start_pipeline(use_synthetic_data=True)

        return True

    except KeyboardInterrupt:
        logger.info("Pipeline stopped by user")
        return True
    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        return False


def collect_data():
    try:
        collector = NetworkDataCollector()
        data = collector.generate_synthetic_data(1000)

        output_file = Path("data") / "captured" / "sample_traffic.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        data.to_csv(output_file, index=False)
        logger.info(f"Data saved to {output_file}")

        print(f"Total samples: {len(data)}")
        print(f"Anomalies: {data['is_anomaly'].sum()}")
        print(f"Anomaly rate: {data['is_anomaly'].mean()*100:.2f}%")
        print(f"Unique source IPs: {data['src_ip'].nunique()}")
        print(f"Protocols: {', '.join(data['protocol'].unique())}")

        return True

    except Exception as e:
        logger.error(f"Error collecting data: {e}")
        return False


def run_tests():
    try:
        collector = NetworkDataCollector()
        test_data = collector.generate_synthetic_data(100)
        assert not test_data.empty, "Data collection failed"
        logger.info("Data collection test passed")

        processor = NetworkDataProcessor()
        processed_data = processor.parse_cicids2017_format(test_data)
        assert not processed_data.empty, "Data processing failed"
        logger.info("Data processing test passed")

        detector = AnomalyDetector()
        X, y = processor.prepare_features_for_ml(processed_data)
        assert not X.empty, "Feature preparation failed"

        results = detector.train_isolation_forest(X)
        assert 'predictions' in results, "Model training failed"
        logger.info("Anomaly detection test passed")

        logger.info("All tests passed")
        return True

    except Exception as e:
        logger.error(f"Test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Network Traffic Anomaly Detection System")

    parser.add_argument(
        'command',
        choices=['train', 'dashboard', 'pipeline', 'collect', 'test'],
        help='Command to run'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    success = False

    if args.command == 'train':
        success = train_models()
    elif args.command == 'dashboard':
        success = run_dashboard()
    elif args.command == 'pipeline':
        success = run_pipeline()
    elif args.command == 'collect':
        success = collect_data()
    elif args.command == 'test':
        success = run_tests()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
