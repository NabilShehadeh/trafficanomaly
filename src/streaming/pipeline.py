"""
Real-Time Streaming Pipeline for Network Traffic Anomaly Detection

This module implements:
1. Kafka producer for streaming network data
2. Spark streaming consumer for real-time processing
3. Real-time anomaly detection pipeline
4. Alert generation and notification system
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import threading
import queue

# Kafka Libraries
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
    KAFKA_AVAILABLE = True
except ImportError:
    KAFKA_AVAILABLE = False
    print("Warning: Kafka not available. Install with: pip install kafka-python")

# Spark Libraries
try:
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import col, from_json, window, count, avg, max, min
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType
    from pyspark.streaming import StreamingContext
    from pyspark.streaming.kafka import KafkaUtils
    SPARK_AVAILABLE = True
except ImportError:
    SPARK_AVAILABLE = False
    print("Warning: PySpark not available. Install with: pip install pyspark")

# Import our custom modules
import sys
sys.path.append(str(Path(__file__).parent.parent))
from data_collection.collector import NetworkDataCollector
from data_processing.processor import NetworkDataProcessor
from models.anomaly_detector import AnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaProducerService:
    """Kafka producer for streaming network traffic data"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "network-traffic"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None
        self.data_collector = NetworkDataCollector()
        self.is_running = False
        
    def initialize_producer(self) -> bool:
        """Initialize Kafka producer"""
        if not KAFKA_AVAILABLE:
            logger.error("Kafka not available")
            return False
        
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None,
                retries=3,
                acks='all'
            )
            logger.info(f"Kafka producer initialized for topic: {self.topic}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Kafka producer: {e}")
            return False
    
    def send_network_data(self, data: Dict[str, Any]) -> bool:
        """Send network data to Kafka topic"""
        try:
            if not self.producer:
                logger.error("Producer not initialized")
                return False
            
            # Add timestamp
            data['timestamp'] = datetime.now().isoformat()
            
            # Send to Kafka
            future = self.producer.send(self.topic, value=data)
            record_metadata = future.get(timeout=10)
            
            logger.debug(f"Data sent to topic {record_metadata.topic}, partition {record_metadata.partition}")
            return True
            
        except KafkaError as e:
            logger.error(f"Kafka error: {e}")
            return False
        except Exception as e:
            logger.error(f"Error sending data: {e}")
            return False
    
    def stream_live_traffic(self, interface: str = None, packet_count: int = 100) -> None:
        """Stream live network traffic to Kafka"""
        try:
            logger.info("Starting live traffic streaming...")
            self.is_running = True
            
            while self.is_running:
                # Capture packets
                packets = self.data_collector.capture_live_traffic(interface, packet_count)
                
                # Send each packet to Kafka
                for packet in packets:
                    if self.send_network_data(packet):
                        logger.debug(f"Sent packet: {packet.get('src_ip', 'unknown')} -> {packet.get('dst_ip', 'unknown')}")
                
                # Wait before next capture
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Stopping live traffic streaming...")
            self.is_running = False
        except Exception as e:
            logger.error(f"Error in live traffic streaming: {e}")
            self.is_running = False
    
    def stream_synthetic_data(self, interval: float = 1.0) -> None:
        """Stream synthetic network data for testing"""
        try:
            logger.info("Starting synthetic data streaming...")
            self.is_running = True
            
            while self.is_running:
                # Generate synthetic packet
                synthetic_packet = self._generate_synthetic_packet()
                
                if self.send_network_data(synthetic_packet):
                    logger.debug(f"Sent synthetic packet: {synthetic_packet['src_ip']} -> {synthetic_packet['dst_ip']}")
                
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Stopping synthetic data streaming...")
            self.is_running = False
        except Exception as e:
            logger.error(f"Error in synthetic data streaming: {e}")
            self.is_running = False
    
    def _generate_synthetic_packet(self) -> Dict[str, Any]:
        """Generate synthetic network packet data"""
        np.random.seed(int(time.time()))
        
        # Randomly decide if this is an anomaly
        is_anomaly = np.random.random() < 0.05  # 5% chance of anomaly
        
        if is_anomaly:
            # Generate anomalous packet
            packet = {
                'src_ip': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'dst_ip': f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'src_port': np.random.randint(1, 65535),
                'dst_port': np.random.randint(1, 1024),  # Well-known ports
                'protocol': np.random.choice(['TCP', 'UDP', 'ICMP']),
                'packet_length': np.random.randint(1500, 9000),  # Large packets
                'duration': np.random.exponential(0.5),  # Longer duration
                'flow_bytes_sent': np.random.randint(10000, 100000),
                'flow_bytes_received': np.random.randint(1000, 10000),
                'packets_sent': np.random.randint(100, 1000),
                'packets_received': np.random.randint(10, 100),
                'is_anomaly': 1
            }
        else:
            # Generate normal packet
            packet = {
                'src_ip': f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}",
                'dst_ip': f"8.8.8.{np.random.randint(1,10)}",  # Common destination
                'src_port': np.random.randint(1024, 65535),
                'dst_port': np.random.choice([80, 443, 53, 22, 21]),  # Common ports
                'protocol': np.random.choice(['TCP', 'UDP'], p=[0.8, 0.2]),
                'packet_length': np.random.randint(64, 1500),
                'duration': np.random.exponential(0.1),
                'flow_bytes_sent': np.random.randint(100, 10000),
                'flow_bytes_received': np.random.randint(100, 10000),
                'packets_sent': np.random.randint(1, 100),
                'packets_received': np.random.randint(1, 100),
                'is_anomaly': 0
            }
        
        return packet
    
    def close(self):
        """Close Kafka producer"""
        if self.producer:
            self.producer.close()
            logger.info("Kafka producer closed")

class SparkStreamingConsumer:
    """Spark streaming consumer for real-time anomaly detection"""
    
    def __init__(self, bootstrap_servers: str = "localhost:9092", topic: str = "network-traffic"):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.spark = None
        self.processor = NetworkDataProcessor()
        self.anomaly_detector = AnomalyDetector()
        self.alert_queue = queue.Queue()
        
    def initialize_spark(self) -> bool:
        """Initialize Spark session"""
        if not SPARK_AVAILABLE:
            logger.error("Spark not available")
            return False
        
        try:
            self.spark = SparkSession.builder \
                .appName("NetworkTrafficAnomalyDetection") \
                .config("spark.sql.adaptive.enabled", "true") \
                .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
                .getOrCreate()
            
            logger.info("Spark session initialized")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Spark: {e}")
            return False
    
    def define_schema(self) -> StructType:
        """Define schema for network traffic data"""
        return StructType([
            StructField("src_ip", StringType(), True),
            StructField("dst_ip", StringType(), True),
            StructField("src_port", IntegerType(), True),
            StructField("dst_port", IntegerType(), True),
            StructField("protocol", StringType(), True),
            StructField("packet_length", IntegerType(), True),
            StructField("duration", DoubleType(), True),
            StructField("flow_bytes_sent", IntegerType(), True),
            StructField("flow_bytes_received", IntegerType(), True),
            StructField("packets_sent", IntegerType(), True),
            StructField("packets_received", IntegerType(), True),
            StructField("is_anomaly", IntegerType(), True),
            StructField("timestamp", StringType(), True)
        ])
    
    def process_streaming_data(self, batch_interval: int = 10) -> None:
        """Process streaming data for anomaly detection"""
        try:
            logger.info("Starting Spark streaming processing...")
            
            # Define schema
            schema = self.define_schema()
            
            # Create streaming DataFrame
            df = self.spark \
                .readStream \
                .format("kafka") \
                .option("kafka.bootstrap.servers", self.bootstrap_servers) \
                .option("subscribe", self.topic) \
                .option("startingOffsets", "latest") \
                .load()
            
            # Parse JSON data
            df_parsed = df.select(
                from_json(col("value").cast("string"), schema).alias("data")
            ).select("data.*")
            
            # Add processing timestamp
            df_with_timestamp = df_parsed.withColumn(
                "processing_time", 
                col("timestamp").cast(TimestampType())
            )
            
            # Calculate windowed aggregations
            windowed_df = df_with_timestamp \
                .withWatermark("processing_time", "1 minute") \
                .groupBy(
                    window(col("processing_time"), "30 seconds"),
                    col("src_ip"),
                    col("dst_ip")
                ) \
                .agg(
                    count("*").alias("packet_count"),
                    avg("packet_length").alias("avg_packet_length"),
                    max("packet_length").alias("max_packet_length"),
                    min("packet_length").alias("min_packet_length"),
                    avg("duration").alias("avg_duration"),
                    sum("flow_bytes_sent").alias("total_bytes_sent"),
                    sum("flow_bytes_received").alias("total_bytes_received")
                )
            
            # Define query
            query = windowed_df \
                .writeStream \
                .outputMode("update") \
                .foreachBatch(self._process_batch) \
                .start()
            
            # Wait for termination
            query.awaitTermination()
            
        except Exception as e:
            logger.error(f"Error in streaming processing: {e}")
    
    def _process_batch(self, batch_df, batch_id):
        """Process each batch of streaming data"""
        try:
            logger.info(f"Processing batch {batch_id} with {batch_df.count()} records")
            
            # Convert to Pandas for processing
            pandas_df = batch_df.toPandas()
            
            if pandas_df.empty:
                return
            
            # Process the data
            processed_df = self.processor.parse_cicids2017_format(pandas_df)
            
            # Prepare features for anomaly detection
            X, _ = self.processor.prepare_features_for_ml(processed_df)
            
            if not X.empty:
                # Detect anomalies using ensemble model
                predictions = self.anomaly_detector.predict_anomalies(X, 'ensemble')
                
                if predictions and 'predictions' in predictions:
                    anomalies = predictions['predictions']
                    
                    # Generate alerts for detected anomalies
                    self._generate_alerts(processed_df, anomalies, batch_id)
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
    
    def _generate_alerts(self, df: pd.DataFrame, anomalies: np.ndarray, batch_id: int):
        """Generate alerts for detected anomalies"""
        try:
            anomaly_indices = np.where(anomalies == 1)[0]
            
            for idx in anomaly_indices:
                if idx < len(df):
                    alert = {
                        'batch_id': batch_id,
                        'timestamp': datetime.now().isoformat(),
                        'alert_type': 'Network Anomaly Detected',
                        'severity': 'HIGH',
                        'src_ip': df.iloc[idx].get('src_ip', 'Unknown'),
                        'dst_ip': df.iloc[idx].get('dst_ip', 'Unknown'),
                        'protocol': df.iloc[idx].get('protocol', 'Unknown'),
                        'packet_length': df.iloc[idx].get('packet_length', 0),
                        'description': 'Unusual network traffic pattern detected',
                        'recommendation': 'Investigate source IP and review network logs'
                    }
                    
                    self.alert_queue.put(alert)
                    logger.warning(f"ALERT: {alert['alert_type']} - {alert['src_ip']} -> {alert['dst_ip']}")
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
    
    def get_alerts(self) -> List[Dict[str, Any]]:
        """Get pending alerts"""
        alerts = []
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                alerts.append(alert)
            except queue.Empty:
                break
        return alerts
    
    def close(self):
        """Close Spark session"""
        if self.spark:
            self.spark.stop()
            logger.info("Spark session closed")

class RealTimePipeline:
    """Main class for real-time anomaly detection pipeline"""
    
    def __init__(self, kafka_servers: str = "localhost:9092", topic: str = "network-traffic"):
        self.kafka_servers = kafka_servers
        self.topic = topic
        self.producer_service = KafkaProducerService(kafka_servers, topic)
        self.consumer_service = SparkStreamingConsumer(kafka_servers, topic)
        self.is_running = False
        
    def start_pipeline(self, use_synthetic_data: bool = True, interface: str = None) -> None:
        """Start the real-time pipeline"""
        try:
            logger.info("Starting real-time anomaly detection pipeline...")
            
            # Initialize services
            if not self.producer_service.initialize_producer():
                logger.error("Failed to initialize producer")
                return
            
            if not self.consumer_service.initialize_spark():
                logger.error("Failed to initialize Spark")
                return
            
            self.is_running = True
            
            # Start producer in separate thread
            if use_synthetic_data:
                producer_thread = threading.Thread(
                    target=self.producer_service.stream_synthetic_data
                )
            else:
                producer_thread = threading.Thread(
                    target=self.producer_service.stream_live_traffic,
                    args=(interface, 100)
                )
            
            producer_thread.daemon = True
            producer_thread.start()
            
            # Start consumer (blocking)
            consumer_thread = threading.Thread(
                target=self.consumer_service.process_streaming_data
            )
            consumer_thread.daemon = True
            consumer_thread.start()
            
            # Monitor alerts
            self._monitor_alerts()
            
        except KeyboardInterrupt:
            logger.info("Stopping pipeline...")
            self.stop_pipeline()
        except Exception as e:
            logger.error(f"Error in pipeline: {e}")
            self.stop_pipeline()
    
    def _monitor_alerts(self):
        """Monitor and display alerts"""
        try:
            while self.is_running:
                alerts = self.consumer_service.get_alerts()
                
                for alert in alerts:
                    logger.warning(f"🚨 ALERT: {alert['alert_type']}")
                    logger.warning(f"   Source: {alert['src_ip']} -> {alert['dst_ip']}")
                    logger.warning(f"   Protocol: {alert['protocol']}")
                    logger.warning(f"   Severity: {alert['severity']}")
                    logger.warning(f"   Time: {alert['timestamp']}")
                    logger.warning(f"   Description: {alert['description']}")
                    logger.warning("   " + "="*50)
                
                time.sleep(5)  # Check every 5 seconds
                
        except Exception as e:
            logger.error(f"Error monitoring alerts: {e}")
    
    def stop_pipeline(self):
        """Stop the pipeline"""
        self.is_running = False
        self.producer_service.is_running = False
        self.producer_service.close()
        self.consumer_service.close()
        logger.info("Pipeline stopped")

def main():
    """Example usage of the real-time pipeline"""
    pipeline = RealTimePipeline()
    
    print("Starting real-time network traffic anomaly detection pipeline...")
    print("Press Ctrl+C to stop")
    
    try:
        # Start with synthetic data for testing
        pipeline.start_pipeline(use_synthetic_data=True)
    except KeyboardInterrupt:
        print("\nStopping pipeline...")
        pipeline.stop_pipeline()

if __name__ == "__main__":
    main()
