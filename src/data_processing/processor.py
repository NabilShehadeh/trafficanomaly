"""
Data Processing Module for Network Traffic Anomaly Detection

This module handles:
1. Parsing structured connection logs (CICIDS2017 format)
2. Processing semi-structured packet captures
3. Feature engineering and data transformation
4. Data validation and cleaning
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import re
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkDataProcessor:
    """Main class for processing network traffic data"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_selector = None
        self.feature_names = []
        
    def parse_cicids2017_format(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse CICIDS2017 dataset format and extract relevant features
        
        Args:
            df: Raw CICIDS2017 DataFrame
            
        Returns:
            pd.DataFrame: Processed DataFrame with engineered features
        """
        try:
            logger.info("Parsing CICIDS2017 format...")
            
            # Create a copy to avoid modifying original
            processed_df = df.copy()
            
            # Handle missing values
            processed_df = self._handle_missing_values(processed_df)
            
            # Extract time-based features
            processed_df = self._extract_time_features(processed_df)
            
            # Extract network flow features
            processed_df = self._extract_flow_features(processed_df)
            
            # Extract statistical features
            processed_df = self._extract_statistical_features(processed_df)
            
            # Extract protocol-specific features
            processed_df = self._extract_protocol_features(processed_df)
            
            # Encode categorical variables
            processed_df = self._encode_categorical_features(processed_df)
            
            logger.info(f"Successfully parsed CICIDS2017 format. Shape: {processed_df.shape}")
            return processed_df
            
        except Exception as e:
            logger.error(f"Error parsing CICIDS2017 format: {e}")
            return df
    
    def parse_packet_capture(self, packets: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Parse raw packet capture data into structured format
        
        Args:
            packets: List of raw packet dictionaries
            
        Returns:
            pd.DataFrame: Structured packet data
        """
        try:
            logger.info(f"Parsing {len(packets)} packets...")
            
            if not packets:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(packets)
            
            # Extract packet-level features
            df = self._extract_packet_features(df)
            
            # Group by flow (src_ip, dst_ip, src_port, dst_port)
            df = self._group_by_flow(df)
            
            # Calculate flow statistics
            df = self._calculate_flow_statistics(df)
            
            logger.info(f"Successfully parsed packet capture. Shape: {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing packet capture: {e}")
            return pd.DataFrame()
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        try:
            # For numeric columns, fill with median
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            for col in numeric_columns:
                if df[col].isnull().any():
                    df[col].fillna(df[col].median(), inplace=True)
            
            # For categorical columns, fill with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                if df[col].isnull().any():
                    mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                    df[col].fillna(mode_value, inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return df
    
    def _extract_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract time-based features"""
        try:
            # If timestamp column exists, extract time features
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df['hour'] = df['timestamp'].dt.hour
                df['day_of_week'] = df['timestamp'].dt.dayofweek
                df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            
            # If duration column exists, extract duration features
            if 'duration' in df.columns:
                df['duration_log'] = np.log1p(df['duration'])
                df['duration_sqrt'] = np.sqrt(df['duration'])
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting time features: {e}")
            return df
    
    def _extract_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract network flow features"""
        try:
            # Calculate flow ratios
            if 'flow_bytes_sent' in df.columns and 'flow_bytes_received' in df.columns:
                df['flow_ratio'] = df['flow_bytes_sent'] / (df['flow_bytes_received'] + 1)
                df['total_flow_bytes'] = df['flow_bytes_sent'] + df['flow_bytes_received']
            
            # Calculate packet ratios
            if 'packets_sent' in df.columns and 'packets_received' in df.columns:
                df['packet_ratio'] = df['packets_sent'] / (df['packets_received'] + 1)
                df['total_packets'] = df['packets_sent'] + df['packets_received']
            
            # Calculate bytes per packet
            if 'total_flow_bytes' in df.columns and 'total_packets' in df.columns:
                df['bytes_per_packet'] = df['total_flow_bytes'] / (df['total_packets'] + 1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting flow features: {e}")
            return df
    
    def _extract_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract statistical features"""
        try:
            # Calculate statistical measures for numeric columns
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            
            for col in numeric_columns:
                if col not in ['is_anomaly', 'label']:  # Skip target variables
                    # Calculate rolling statistics (if enough data)
                    if len(df) > 10:
                        df[f'{col}_zscore'] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)
                        df[f'{col}_percentile'] = df[col].rank(pct=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting statistical features: {e}")
            return df
    
    def _extract_protocol_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract protocol-specific features"""
        try:
            # Protocol encoding
            if 'protocol' in df.columns:
                protocol_mapping = {
                    'TCP': 1, 'UDP': 2, 'ICMP': 3, 'HTTP': 4, 'HTTPS': 5,
                    'FTP': 6, 'SSH': 7, 'DNS': 8, 'DHCP': 9, 'Unknown': 0
                }
                df['protocol_encoded'] = df['protocol'].map(protocol_mapping).fillna(0)
            
            # Port-based features
            if 'dst_port' in df.columns:
                df['is_well_known_port'] = df['dst_port'].isin([21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995]).astype(int)
                df['is_privileged_port'] = (df['dst_port'] < 1024).astype(int)
                df['is_ephemeral_port'] = ((df['dst_port'] >= 49152) & (df['dst_port'] <= 65535)).astype(int)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting protocol features: {e}")
            return df
    
    def _extract_packet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from individual packets"""
        try:
            # Calculate packet inter-arrival times
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
                df = df.sort_values('timestamp')
                df['inter_arrival_time'] = df['timestamp'].diff().dt.total_seconds().fillna(0)
            
            # Extract IP address features
            if 'src_ip' in df.columns:
                df['src_ip_class'] = df['src_ip'].apply(self._get_ip_class)
                df['is_private_src'] = df['src_ip'].apply(self._is_private_ip)
            
            if 'dst_ip' in df.columns:
                df['dst_ip_class'] = df['dst_ip'].apply(self._get_ip_class)
                df['is_private_dst'] = df['dst_ip'].apply(self._is_private_ip)
            
            return df
            
        except Exception as e:
            logger.error(f"Error extracting packet features: {e}")
            return df
    
    def _group_by_flow(self, df: pd.DataFrame) -> pd.DataFrame:
        """Group packets by flow and calculate flow statistics"""
        try:
            if df.empty:
                return df
            
            # Define flow grouping columns
            flow_columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
            available_flow_columns = [col for col in flow_columns if col in df.columns]
            
            if not available_flow_columns:
                return df
            
            # Group by flow and calculate statistics
            flow_stats = df.groupby(available_flow_columns).agg({
                'packet_length': ['count', 'sum', 'mean', 'std', 'min', 'max'],
                'inter_arrival_time': ['mean', 'std', 'min', 'max'],
                'timestamp': ['min', 'max']
            }).reset_index()
            
            # Flatten column names
            flow_stats.columns = ['_'.join(col).strip() if col[1] else col[0] for col in flow_stats.columns]
            
            # Calculate flow duration
            if 'timestamp_min' in flow_stats.columns and 'timestamp_max' in flow_stats.columns:
                flow_stats['flow_duration'] = (pd.to_datetime(flow_stats['timestamp_max']) - 
                                             pd.to_datetime(flow_stats['timestamp_min'])).dt.total_seconds()
            
            return flow_stats
            
        except Exception as e:
            logger.error(f"Error grouping by flow: {e}")
            return df
    
    def _calculate_flow_statistics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate additional flow statistics"""
        try:
            # Calculate flow rates
            if 'packet_length_count' in df.columns and 'flow_duration' in df.columns:
                df['packets_per_second'] = df['packet_length_count'] / (df['flow_duration'] + 1e-8)
                df['bytes_per_second'] = df['packet_length_sum'] / (df['flow_duration'] + 1e-8)
            
            # Calculate flow patterns
            if 'packet_length_mean' in df.columns and 'packet_length_std' in df.columns:
                df['packet_size_variation'] = df['packet_length_std'] / (df['packet_length_mean'] + 1e-8)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating flow statistics: {e}")
            return df
    
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical features"""
        try:
            categorical_columns = df.select_dtypes(include=['object']).columns
            
            for col in categorical_columns:
                if col not in ['src_ip', 'dst_ip']:  # Skip IP addresses for now
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                    else:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
            
            return df
            
        except Exception as e:
            logger.error(f"Error encoding categorical features: {e}")
            return df
    
    def _get_ip_class(self, ip: str) -> str:
        """Get IP address class"""
        try:
            parts = ip.split('.')
            if len(parts) == 4:
                first_octet = int(parts[0])
                if 1 <= first_octet <= 126:
                    return 'A'
                elif 128 <= first_octet <= 191:
                    return 'B'
                elif 192 <= first_octet <= 223:
                    return 'C'
                elif 224 <= first_octet <= 239:
                    return 'D'
                elif 240 <= first_octet <= 255:
                    return 'E'
            return 'Unknown'
        except:
            return 'Unknown'
    
    def _is_private_ip(self, ip: str) -> bool:
        """Check if IP address is private"""
        try:
            parts = ip.split('.')
            if len(parts) == 4:
                first_octet = int(parts[0])
                second_octet = int(parts[1])
                
                # Private IP ranges
                if first_octet == 10:
                    return True
                elif first_octet == 172 and 16 <= second_octet <= 31:
                    return True
                elif first_octet == 192 and second_octet == 168:
                    return True
            return False
        except:
            return False
    
    def prepare_features_for_ml(self, df: pd.DataFrame, target_column: str = 'is_anomaly') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning
        
        Args:
            df: Processed DataFrame
            target_column: Name of target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        try:
            logger.info("Preparing features for machine learning...")
            
            # Select numeric features only
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Remove target column from features
            if target_column in feature_columns:
                feature_columns.remove(target_column)
            
            # Remove other non-feature columns
            exclude_columns = ['timestamp', 'src_ip', 'dst_ip', 'label']
            feature_columns = [col for col in feature_columns if col not in exclude_columns]
            
            # Extract features and target
            X = df[feature_columns].copy()
            y = df[target_column] if target_column in df.columns else None
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], np.nan)
            X = X.fillna(X.median())
            
            # Scale features
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
            
            self.feature_names = feature_columns
            
            logger.info(f"Prepared {X_scaled.shape[1]} features for ML")
            return X_scaled, y
            
        except Exception as e:
            logger.error(f"Error preparing features for ML: {e}")
            return pd.DataFrame(), pd.Series()
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, k: int = 20) -> pd.DataFrame:
        """
        Select top k features using statistical tests
        
        Args:
            X: Feature matrix
            y: Target vector
            k: Number of features to select
            
        Returns:
            pd.DataFrame: Selected features
        """
        try:
            logger.info(f"Selecting top {k} features...")
            
            self.feature_selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
            X_selected = self.feature_selector.fit_transform(X, y)
            
            # Get selected feature names
            selected_features = X.columns[self.feature_selector.get_support()].tolist()
            
            logger.info(f"Selected features: {selected_features}")
            
            return pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            
        except Exception as e:
            logger.error(f"Error selecting features: {e}")
            return X

def main():
    """Example usage of NetworkDataProcessor"""
    processor = NetworkDataProcessor()
    
    # Create sample data
    sample_data = {
        'src_ip': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
        'dst_ip': ['8.8.8.8', '1.1.1.1', '208.67.222.222'],
        'src_port': [12345, 54321, 8080],
        'dst_port': [80, 443, 53],
        'protocol': ['TCP', 'UDP', 'TCP'],
        'packet_length': [1500, 512, 1024],
        'duration': [0.1, 0.05, 0.2],
        'is_anomaly': [0, 1, 0]
    }
    
    df = pd.DataFrame(sample_data)
    print("Original data:")
    print(df)
    
    # Process the data
    processed_df = processor.parse_cicids2017_format(df)
    print("\nProcessed data:")
    print(processed_df.head())
    
    # Prepare for ML
    X, y = processor.prepare_features_for_ml(processed_df)
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")

if __name__ == "__main__":
    main()
