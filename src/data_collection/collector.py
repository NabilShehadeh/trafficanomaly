"""
Data Collection Module for Network Traffic Anomaly Detection

This module handles:
1. CICIDS2017 dataset integration
2. Live packet capture via Wireshark/Scapy
3. Network traffic data preprocessing
"""

import os
import pandas as pd
import numpy as np
import requests
import zipfile
from typing import Optional, List, Dict, Any
import logging
from pathlib import Path

# Network packet capture
try:
    import scapy.all as scapy
    from scapy.layers.inet import IP, TCP, UDP, ICMP
    from scapy.layers.l2 import Ether
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    print("Warning: Scapy not available. Install with: pip install scapy")

# Packet analysis
try:
    import pyshark
    PYSHARK_AVAILABLE = True
except ImportError:
    PYSHARK_AVAILABLE = False
    print("Warning: PyShark not available. Install with: pip install pyshark")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkDataCollector:
    """Main class for collecting network traffic data"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.cicids_dir = self.data_dir / "cicids2017"
        self.captured_dir = self.data_dir / "captured"
        
        # Create directories
        self.cicids_dir.mkdir(parents=True, exist_ok=True)
        self.captured_dir.mkdir(parents=True, exist_ok=True)
    
    def download_cicids2017(self, force_download: bool = False) -> bool:
        """
        Download CICIDS2017 dataset
        
        Args:
            force_download: Whether to force re-download even if files exist
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # CICIDS2017 dataset URLs (these are example URLs - actual URLs may vary)
            dataset_urls = {
                "Monday": "https://www.unb.ca/cic/datasets/ids-2017.html",
                "Tuesday": "https://www.unb.ca/cic/datasets/ids-2017.html", 
                "Wednesday": "https://www.unb.ca/cic/datasets/ids-2017.html",
                "Thursday": "https://www.unb.ca/cic/datasets/ids-2017.html",
                "Friday": "https://www.unb.ca/cic/datasets/ids-2017.html"
            }
            
            logger.info("CICIDS2017 dataset download initiated")
            logger.info("Note: Please download the dataset manually from:")
            logger.info("https://www.unb.ca/cic/datasets/ids-2017.html")
            logger.info("Place the CSV files in the data/cicids2017/ directory")
            
            return True
            
        except Exception as e:
            logger.error(f"Error downloading CICIDS2017 dataset: {e}")
            return False
    
    def load_cicids2017_data(self, day: str = "Monday") -> Optional[pd.DataFrame]:
        """
        Load CICIDS2017 data for a specific day
        
        Args:
            day: Day of the week (Monday, Tuesday, etc.)
            
        Returns:
            pd.DataFrame: Loaded dataset or None if error
        """
        try:
            csv_file = self.cicids_dir / f"{day}-WorkingHours.pcap_ISCX.csv"
            
            if not csv_file.exists():
                logger.error(f"CICIDS2017 {day} data not found. Please download first.")
                return None
            
            logger.info(f"Loading CICIDS2017 {day} data...")
            df = pd.read_csv(csv_file)
            
            # Basic preprocessing
            df = self._preprocess_cicids_data(df)
            
            logger.info(f"Loaded {len(df)} records from CICIDS2017 {day}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CICIDS2017 data: {e}")
            return None
    
    def _preprocess_cicids_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess CICIDS2017 data"""
        try:
            # Remove infinite values
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Fill missing values
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())
            
            # Convert categorical columns
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df[col] = df[col].fillna('Unknown')
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing CICIDS2017 data: {e}")
            return df
    
    def capture_live_traffic(self, interface: str = None, count: int = 1000) -> List[Dict[str, Any]]:
        """
        Capture live network traffic using Scapy
        
        Args:
            interface: Network interface to capture on (None for default)
            count: Number of packets to capture
            
        Returns:
            List[Dict]: List of parsed packet data
        """
        if not SCAPY_AVAILABLE:
            logger.error("Scapy not available for live traffic capture")
            return []
        
        try:
            logger.info(f"Capturing {count} packets from interface {interface or 'default'}")
            
            # Capture packets
            packets = scapy.sniff(iface=interface, count=count)
            
            # Parse packets
            parsed_packets = []
            for packet in packets:
                parsed_data = self._parse_packet(packet)
                if parsed_data:
                    parsed_packets.append(parsed_data)
            
            logger.info(f"Successfully captured and parsed {len(parsed_packets)} packets")
            return parsed_packets
            
        except Exception as e:
            logger.error(f"Error capturing live traffic: {e}")
            return []
    
    def _parse_packet(self, packet) -> Optional[Dict[str, Any]]:
        """Parse a single packet into structured data"""
        try:
            parsed_data = {
                'timestamp': packet.time,
                'protocol': packet.proto if hasattr(packet, 'proto') else 'Unknown',
                'length': len(packet)
            }
            
            # Parse IP layer
            if packet.haslayer(IP):
                ip_layer = packet[IP]
                parsed_data.update({
                    'src_ip': ip_layer.src,
                    'dst_ip': ip_layer.dst,
                    'ip_version': ip_layer.version,
                    'ip_len': ip_layer.len,
                    'ip_ttl': ip_layer.ttl
                })
            
            # Parse TCP layer
            if packet.haslayer(TCP):
                tcp_layer = packet[TCP]
                parsed_data.update({
                    'src_port': tcp_layer.sport,
                    'dst_port': tcp_layer.dport,
                    'tcp_flags': tcp_layer.flags,
                    'tcp_seq': tcp_layer.seq,
                    'tcp_ack': tcp_layer.ack
                })
            
            # Parse UDP layer
            elif packet.haslayer(UDP):
                udp_layer = packet[UDP]
                parsed_data.update({
                    'src_port': udp_layer.sport,
                    'dst_port': udp_layer.dport,
                    'udp_len': udp_layer.len
                })
            
            # Parse ICMP layer
            elif packet.haslayer(ICMP):
                icmp_layer = packet[ICMP]
                parsed_data.update({
                    'icmp_type': icmp_layer.type,
                    'icmp_code': icmp_layer.code
                })
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing packet: {e}")
            return None
    
    def save_captured_data(self, packets: List[Dict[str, Any]], filename: str) -> bool:
        """Save captured packet data to file"""
        try:
            filepath = self.captured_dir / f"{filename}.csv"
            df = pd.DataFrame(packets)
            df.to_csv(filepath, index=False)
            logger.info(f"Saved {len(packets)} packets to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving captured data: {e}")
            return False
    
    def generate_synthetic_data(self, num_samples: int = 10000) -> pd.DataFrame:
        """
        Generate synthetic network traffic data for testing
        
        Args:
            num_samples: Number of samples to generate
            
        Returns:
            pd.DataFrame: Generated synthetic data
        """
        try:
            logger.info(f"Generating {num_samples} synthetic network traffic samples")
            
            # Generate synthetic features similar to CICIDS2017
            np.random.seed(42)
            
            data = {
                'src_ip': [f"192.168.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(num_samples)],
                'dst_ip': [f"10.0.{np.random.randint(1,255)}.{np.random.randint(1,255)}" for _ in range(num_samples)],
                'src_port': np.random.randint(1, 65535, num_samples),
                'dst_port': np.random.randint(1, 65535, num_samples),
                'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], num_samples),
                'packet_length': np.random.randint(64, 1500, num_samples),
                'duration': np.random.exponential(0.1, num_samples),
                'flow_bytes_sent': np.random.randint(100, 10000, num_samples),
                'flow_bytes_received': np.random.randint(100, 10000, num_samples),
                'packets_sent': np.random.randint(1, 100, num_samples),
                'packets_received': np.random.randint(1, 100, num_samples),
                'is_anomaly': np.random.choice([0, 1], num_samples, p=[0.95, 0.05])  # 5% anomalies
            }
            
            df = pd.DataFrame(data)
            logger.info(f"Generated synthetic dataset with {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error generating synthetic data: {e}")
            return pd.DataFrame()

def main():
    """Example usage of NetworkDataCollector"""
    collector = NetworkDataCollector()
    
    # Generate synthetic data for testing
    synthetic_data = collector.generate_synthetic_data(1000)
    print(f"Generated {len(synthetic_data)} synthetic samples")
    print(synthetic_data.head())
    
    # Try to capture live traffic (requires root privileges)
    # live_packets = collector.capture_live_traffic(count=100)
    # print(f"Captured {len(live_packets)} live packets")

if __name__ == "__main__":
    main()
