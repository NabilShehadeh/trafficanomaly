# Deployment Guide

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
   az container create --resource-group myResourceGroup \
     --name anomaly-detection \
     --image network-anomaly-detection \
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
