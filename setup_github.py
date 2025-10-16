#!/usr/bin/env python3
"""
GitHub Setup Script for Network Traffic Anomaly Detection System

This script helps set up the project for GitHub deployment and collaboration.
"""

import os
import sys
import subprocess
import json
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

def create_github_workflows():
    """Create GitHub Actions workflows"""
    print("🔧 Creating GitHub Actions workflows...")
    
    workflows_dir = Path(".github/workflows")
    workflows_dir.mkdir(parents=True, exist_ok=True)
    
    # CI/CD workflow
    ci_workflow = """name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.8, 3.9, '3.10']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v3
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        python trafficanomaly.py test
    
    - name: Run linting
      run: |
        pip install flake8 black
        flake8 src/ --count --select=E9,F63,F7,F82 --show-source --statistics
        black --check src/

  build:
    needs: test
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v3
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Build Docker image
      run: |
        docker build -t network-anomaly-detection .
    
    - name: Test Docker container
      run: |
        docker run --rm network-anomaly-detection python trafficanomaly.py test

  deploy:
    needs: [test, build]
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to production
      run: |
        echo "Deployment would happen here"
        # Add your deployment commands here
"""
    
    with open(workflows_dir / "ci-cd.yml", "w") as f:
        f.write(ci_workflow)
    
    print("✅ GitHub Actions workflows created")
    return True

def create_contributing_guide():
    """Create contributing guide"""
    print("📝 Creating contributing guide...")
    
    contributing_content = """# Contributing to Network Traffic Anomaly Detection System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Network Traffic Anomaly Detection System.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/traffic-anomaly-detection.git`
3. Create a new branch: `git checkout -b feature/your-feature-name`
4. Install dependencies: `pip install -r requirements.txt`
5. Run tests: `python trafficanomaly.py test`

## Development Workflow

### Code Style
- Follow PEP 8 style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions small and focused

### Testing
- Write tests for new features
- Ensure all tests pass: `python trafficanomaly.py test`
- Aim for good test coverage

### Documentation
- Update README.md for significant changes
- Add docstrings to new functions/classes
- Update API documentation if needed

## Submitting Changes

1. Make your changes
2. Run tests: `python trafficanomaly.py test`
3. Run linting: `flake8 src/` and `black src/`
4. Commit your changes: `git commit -m "Add feature: description"`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Create a Pull Request

## Pull Request Guidelines

- Provide a clear description of changes
- Reference any related issues
- Ensure CI/CD pipeline passes
- Request review from maintainers

## Issue Reporting

When reporting issues, please include:
- Operating system and Python version
- Steps to reproduce the issue
- Expected vs actual behavior
- Relevant error messages/logs

## Feature Requests

For feature requests, please:
- Describe the feature clearly
- Explain the use case
- Consider implementation complexity
- Check for existing similar requests

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow
- Follow the project's coding standards

Thank you for contributing! 🎉
"""
    
    with open("CONTRIBUTING.md", "w") as f:
        f.write(contributing_content)
    
    print("✅ Contributing guide created")
    return True

def create_license():
    """Create MIT license file"""
    print("📄 Creating license file...")
    
    license_content = """MIT License

Copyright (c) 2024 Network Traffic Anomaly Detection System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    
    with open("LICENSE", "w") as f:
        f.write(license_content)
    
    print("✅ License file created")
    return True

def create_issue_templates():
    """Create GitHub issue templates"""
    print("📋 Creating issue templates...")
    
    templates_dir = Path(".github/ISSUE_TEMPLATE")
    templates_dir.mkdir(parents=True, exist_ok=True)
    
    # Bug report template
    bug_template = """---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment:**
 - OS: [e.g. Ubuntu 20.04, macOS 12.0, Windows 10]
 - Python version: [e.g. 3.9.7]
 - Package versions: [e.g. pandas 1.3.3, scikit-learn 1.0.1]

**Additional context**
Add any other context about the problem here.
"""
    
    with open(templates_dir / "bug_report.md", "w") as f:
        f.write(bug_template)
    
    # Feature request template
    feature_template = """---
name: Feature Request
about: Suggest an idea for this project
title: '[FEATURE] '
labels: enhancement
assignees: ''

---

**Is your feature request related to a problem? Please describe.**
A clear and concise description of what the problem is. Ex. I'm always frustrated when [...]

**Describe the solution you'd like**
A clear and concise description of what you want to happen.

**Describe alternatives you've considered**
A clear and concise description of any alternative solutions or features you've considered.

**Additional context**
Add any other context or screenshots about the feature request here.
"""
    
    with open(templates_dir / "feature_request.md", "w") as f:
        f.write(feature_template)
    
    print("✅ Issue templates created")
    return True

def create_project_structure_doc():
    """Create project structure documentation"""
    print("📚 Creating project structure documentation...")
    
    structure_content = """# Project Structure

This document describes the structure of the Network Traffic Anomaly Detection System.

```
traffic-anomaly-detection/
├── .github/                    # GitHub configuration
│   ├── workflows/              # GitHub Actions workflows
│   └── ISSUE_TEMPLATE/         # Issue templates
├── config/                     # Configuration files
│   └── settings.py             # Main configuration
├── data/                       # Data storage
│   ├── cicids2017/             # CICIDS2017 dataset
│   └── captured/               # Live packet captures
├── models/                     # Trained models
├── logs/                       # Log files
├── src/                        # Source code
│   ├── data_collection/        # Data collection modules
│   │   └── collector.py        # Main data collector
│   ├── data_processing/        # Data processing modules
│   │   └── processor.py        # Data processor
│   ├── models/                 # ML models
│   │   ├── anomaly_detector.py # Anomaly detection models
│   │   └── train_models.py     # Model training script
│   ├── streaming/              # Real-time processing
│   │   └── pipeline.py         # Streaming pipeline
│   └── dashboard/              # Dashboard
│       └── app.py              # Streamlit dashboard
├── tests/                      # Test files
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose setup
├── setup.py                    # Setup script
├── trafficanomaly.py          # Main entry point
├── README.md                   # Project documentation
├── CONTRIBUTING.md             # Contributing guidelines
├── LICENSE                     # MIT License
└── .gitignore                  # Git ignore rules
```

## Key Components

### Data Collection (`src/data_collection/`)
- **collector.py**: Main data collection module
  - CICIDS2017 dataset integration
  - Live packet capture via Scapy
  - Synthetic data generation

### Data Processing (`src/data_processing/`)
- **processor.py**: Data processing and feature engineering
  - CICIDS2017 format parsing
  - Packet capture processing
  - Feature extraction and engineering
  - Data validation and cleaning

### Models (`src/models/`)
- **anomaly_detector.py**: Anomaly detection algorithms
  - Isolation Forest
  - Autoencoder (Deep Learning)
  - K-Means and DBSCAN clustering
  - Ensemble methods
- **train_models.py**: Model training script

### Streaming (`src/streaming/`)
- **pipeline.py**: Real-time processing pipeline
  - Kafka producer for data streaming
  - Spark streaming consumer
  - Real-time anomaly detection
  - Alert generation

### Dashboard (`src/dashboard/`)
- **app.py**: Streamlit dashboard
  - Real-time monitoring
  - Traffic visualizations
  - Alert management
  - Model performance metrics

### Configuration (`config/`)
- **settings.py**: Centralized configuration
  - Kafka and Spark settings
  - Model parameters
  - Environment-specific configs

## Data Flow

1. **Data Collection**: Network traffic data is collected from various sources
2. **Data Processing**: Raw data is parsed and features are engineered
3. **Model Training**: Anomaly detection models are trained on processed data
4. **Real-time Processing**: Live data flows through Kafka and Spark
5. **Anomaly Detection**: Trained models detect anomalies in real-time
6. **Alert Generation**: Alerts are generated for detected anomalies
7. **Dashboard Display**: Results are visualized in the Streamlit dashboard

## Dependencies

### Core Dependencies
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **tensorflow**: Deep learning (for autoencoders)

### Streaming Dependencies
- **kafka-python**: Kafka client
- **pyspark**: Apache Spark integration

### Network Analysis
- **scapy**: Packet manipulation and capture
- **pyshark**: Wireshark integration

### Dashboard
- **streamlit**: Web dashboard framework
- **plotly**: Interactive visualizations

### Optional Dependencies
- **pyod**: Additional anomaly detection algorithms
- **dpkt**: Packet parsing utilities
"""
    
    with open("PROJECT_STRUCTURE.md", "w") as f:
        f.write(structure_content)
    
    print("✅ Project structure documentation created")
    return True

def setup_git_repository():
    """Set up Git repository for GitHub"""
    print("🔧 Setting up Git repository...")
    
    # Initialize Git if not already done
    if not Path(".git").exists():
        run_command("git init", "Initializing Git repository")
    
    # Add all files
    run_command("git add .", "Adding files to Git")
    
    # Create initial commit
    run_command('git commit -m "Initial commit: Complete Network Traffic Anomaly Detection System"', 
                "Creating initial commit")
    
    print("✅ Git repository setup completed")
    return True

def create_deployment_guide():
    """Create deployment guide"""
    print("🚀 Creating deployment guide...")
    
    deployment_content = """# Deployment Guide

This guide covers different deployment options for the Network Traffic Anomaly Detection System.

## Local Development Setup

### Prerequisites
- Python 3.8+
- Git
- Docker (optional)

### Quick Start
```bash
# Clone the repository
git clone <repository-url>
cd traffic-anomaly-detection

# Run setup script
python setup.py

# Train models
python trafficanomaly.py train

# Start dashboard
python trafficanomaly.py dashboard
```

## Docker Deployment

### Using Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Manual Docker Build
```bash
# Build image
docker build -t network-anomaly-detection .

# Run container
docker run -p 8501:8501 network-anomaly-detection
```

## Cloud Deployment

### AWS Deployment
1. **EC2 Instance**:
   ```bash
   # Launch EC2 instance
   # Install Docker
   sudo yum update -y
   sudo yum install -y docker
   sudo service docker start
   
   # Clone and deploy
   git clone <repository-url>
   cd traffic-anomaly-detection
   docker-compose up -d
   ```

2. **AWS ECS**:
   - Create ECS cluster
   - Build and push Docker image to ECR
   - Create ECS service with task definition

### Google Cloud Platform
1. **Compute Engine**:
   ```bash
   # Similar to AWS EC2 setup
   gcloud compute instances create anomaly-detection
   ```

2. **Google Kubernetes Engine (GKE)**:
   ```bash
   # Create GKE cluster
   gcloud container clusters create anomaly-detection
   
   # Deploy using kubectl
   kubectl apply -f k8s/
   ```

### Azure Deployment
1. **Azure Container Instances**:
   ```bash
   # Deploy container
   az container create --resource-group myResourceGroup \\
     --name anomaly-detection \\
     --image network-anomaly-detection \\
     --ports 8501
   ```

## Production Considerations

### Security
- Use HTTPS for dashboard access
- Implement authentication/authorization
- Secure Kafka and Spark clusters
- Regular security updates

### Monitoring
- Set up Prometheus and Grafana
- Monitor system resources
- Set up alerting for anomalies
- Log aggregation and analysis

### Scaling
- Horizontal scaling with multiple Spark workers
- Kafka partitioning for high throughput
- Load balancing for dashboard access
- Database clustering for large datasets

### Backup and Recovery
- Regular model backups
- Data backup strategies
- Disaster recovery procedures
- Configuration management

## Environment Variables

### Required Variables
```bash
ENVIRONMENT=production
KAFKA_BOOTSTRAP_SERVERS=kafka-cluster:9092
SPARK_MASTER=spark://spark-master:7077
```

### Optional Variables
```bash
LOG_LEVEL=INFO
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501
MAX_MEMORY_USAGE=2GB
NUM_WORKERS=4
```

## Troubleshooting

### Common Issues
1. **Kafka Connection Issues**:
   - Check Kafka server status
   - Verify network connectivity
   - Check firewall settings

2. **Spark Job Failures**:
   - Check Spark master/worker status
   - Verify memory allocation
   - Check job logs

3. **Model Training Issues**:
   - Verify data availability
   - Check feature engineering
   - Monitor memory usage

### Log Analysis
```bash
# View application logs
docker-compose logs anomaly-detection

# View Spark logs
docker-compose logs spark-master
docker-compose logs spark-worker

# View Kafka logs
docker-compose logs kafka
```

## Performance Optimization

### System Tuning
- Optimize JVM settings for Spark
- Tune Kafka producer/consumer settings
- Configure appropriate batch sizes
- Monitor resource utilization

### Model Optimization
- Feature selection and engineering
- Model hyperparameter tuning
- Ensemble method optimization
- Real-time inference optimization

## Maintenance

### Regular Tasks
- Update dependencies
- Retrain models periodically
- Monitor system performance
- Clean up old logs and data

### Updates
- Follow semantic versioning
- Test updates in staging environment
- Plan maintenance windows
- Document changes and migrations
"""
    
    with open("DEPLOYMENT.md", "w") as f:
        f.write(deployment_content)
    
    print("✅ Deployment guide created")
    return True

def main():
    """Main GitHub setup function"""
    print("🚀 GitHub Setup for Network Traffic Anomaly Detection System")
    print("=" * 70)
    
    # Create GitHub-specific files
    create_github_workflows()
    create_contributing_guide()
    create_license()
    create_issue_templates()
    create_project_structure_doc()
    create_deployment_guide()
    
    # Setup Git repository
    setup_git_repository()
    
    print("\n" + "=" * 70)
    print("🎉 GitHub setup completed!")
    print("\nNext steps:")
    print("1. Create a new repository on GitHub")
    print("2. Add remote origin:")
    print("   git remote add origin https://github.com/yourusername/traffic-anomaly-detection.git")
    print("3. Push to GitHub:")
    print("   git push -u origin main")
    print("4. Enable GitHub Actions in repository settings")
    print("5. Configure branch protection rules")
    print("6. Set up issue templates and project boards")
    print("\nYour project is now ready for GitHub! 🚀")

if __name__ == "__main__":
    main()
