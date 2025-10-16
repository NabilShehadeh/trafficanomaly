#!/usr/bin/env python3
"""
Setup script for Network Traffic Anomaly Detection System

This script helps set up the environment and dependencies for the project.
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def install_dependencies():
    """Install Python dependencies"""
    print("📦 Installing Python dependencies...")
    
    # Install basic requirements
    if not run_command("pip install -r requirements.txt", "Installing basic requirements"):
        return False
    
    # Install optional dependencies based on platform
    system = platform.system().lower()
    
    if system == "linux":
        print("🐧 Detected Linux system")
        # Install system dependencies for packet capture
        run_command("sudo apt-get update", "Updating package list")
        run_command("sudo apt-get install -y tcpdump wireshark-common", "Installing packet capture tools")
    
    elif system == "darwin":  # macOS
        print("🍎 Detected macOS system")
        # Install via Homebrew if available
        run_command("brew install tcpdump wireshark", "Installing packet capture tools")
    
    elif system == "windows":
        print("🪟 Detected Windows system")
        print("ℹ️  For packet capture on Windows, install WinPcap or Npcap manually")
    
    return True

def setup_directories():
    """Create necessary directories"""
    print("📁 Setting up directories...")
    
    directories = [
        "data/cicids2017",
        "data/captured", 
        "models",
        "logs",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True

def setup_environment_file():
    """Create environment configuration file"""
    print("⚙️ Setting up environment configuration...")
    
    env_content = """# Network Traffic Anomaly Detection Environment Configuration

# Environment (development, production, testing)
ENVIRONMENT=development

# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=network-traffic
KAFKA_GROUP_ID=anomaly-detection-group

# Spark Configuration
SPARK_MASTER=local[*]

# Database Configuration (if using)
DATABASE_URL=sqlite:///data/anomaly_detection.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/anomaly_detection.log

# Dashboard Configuration
DASHBOARD_HOST=localhost
DASHBOARD_PORT=8501

# Security Configuration
SECRET_KEY=your-secret-key-here
ALLOWED_HOSTS=localhost,127.0.0.1

# Performance Configuration
MAX_MEMORY_USAGE=2GB
NUM_WORKERS=4
BATCH_SIZE=1000
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    return True

def setup_git():
    """Initialize Git repository"""
    print("🔧 Setting up Git repository...")
    
    if not Path(".git").exists():
        run_command("git init", "Initializing Git repository")
        run_command("git add .", "Adding files to Git")
        run_command('git commit -m "Initial commit: Network Traffic Anomaly Detection System"', "Creating initial commit")
        print("✅ Git repository initialized")
    else:
        print("ℹ️  Git repository already exists")
    
    return True

def create_gitignore():
    """Create .gitignore file"""
    print("📝 Creating .gitignore file...")
    
    gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# Virtual Environment
venv/
env/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db

# Project specific
data/captured/*.pcap
data/captured/*.csv
logs/*.log
models/*.joblib
models/*.h5
models/*.pkl
.env
*.db
*.sqlite

# Jupyter Notebook
.ipynb_checkpoints

# pytest
.pytest_cache/

# Coverage
htmlcov/
.coverage
.coverage.*
coverage.xml

# Spark
spark-warehouse/
metastore_db/
derby.log

# Kafka
kafka-logs/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    
    print("✅ Created .gitignore file")
    return True

def run_tests():
    """Run basic tests to verify installation"""
    print("🧪 Running basic tests...")
    
    try:
        # Test imports
        import pandas as pd
        import numpy as np
        import sklearn
        print("✅ Core dependencies imported successfully")
        
        # Test our modules
        sys.path.append(str(Path.cwd()))
        from src.data_collection.collector import NetworkDataCollector
        from src.data_processing.processor import NetworkDataProcessor
        from src.models.anomaly_detector import AnomalyDetector
        print("✅ Project modules imported successfully")
        
        # Test basic functionality
        collector = NetworkDataCollector()
        processor = NetworkDataProcessor()
        detector = AnomalyDetector()
        print("✅ Modules instantiated successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def main():
    """Main setup function"""
    print("🛡️ Network Traffic Anomaly Detection System Setup")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Setup directories
    if not setup_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️  Some dependencies failed to install. Continuing...")
    
    # Setup environment
    setup_environment_file()
    
    # Setup Git
    setup_git()
    create_gitignore()
    
    # Run tests
    if not run_tests():
        print("⚠️  Some tests failed. Check your installation.")
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Download CICIDS2017 dataset (optional):")
    print("   python src/data_collection/download_cicids2017.py")
    print("2. Train models:")
    print("   python src/models/train_models.py")
    print("3. Start the dashboard:")
    print("   streamlit run src/dashboard/app.py")
    print("4. Start the streaming pipeline:")
    print("   python src/streaming/pipeline.py")
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
