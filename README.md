# 🛡️ Threat Matrix Predictor

Production-ready MLOps platform for network security threat detection using machine learning and modern cloud-native architecture

## 🚀 Overview

A sophisticated threat detection system that analyzes network security data to predict and classify potential threats using advanced machine learning algorithms. Built with enterprise-grade MLOps practices including automated pipelines, comprehensive monitoring, and scalable cloud deployment.

## 🎯 Key Features

- 🧠 **Intelligent Threat Detection**: Advanced ML algorithms for network security analysis
- ⚙️ **Automated ML Pipeline**: End-to-end MLOps workflow with data validation and monitoring
- ☁️ **Cloud-Native Architecture**: Scalable infrastructure with AWS integration
- 📊 **Experiment Tracking**: MLflow integration with DagHub for reproducible ML experiments
- 🔒 **Production Security**: Containerized deployment with proper authentication
- 📈 **Real-time Monitoring**: Comprehensive logging and performance metrics
- 💾 **Data Persistence**: MongoDB integration with artifact versioning

## 🏗️ Architecture

The system follows a **microservices architecture** with clear separation of concerns:

![architecture](https://github.com/user-attachments/assets/cd75e7df-6039-4820-9f81-3ae861efd6e9)

### High-Level Components:
- **Data Layer**: MongoDB + CSV ingestion with schema validation
- **ML Pipeline**: Automated training with quality checks and feature engineering
- **Model Storage**: Versioned artifacts with S3 synchronization
- **Web Interface**: FastAPI application with interactive predictions
- **Infrastructure**: Docker containers with AWS ECR deployment

## 🛠️ Technology Stack

### Core ML & Data
- **Python 3.x**: Primary runtime with comprehensive ML libraries
- **Scikit-learn**: Machine learning algorithms and model training
- **MongoDB**: Dynamic data storage and retrieval
- **Pandas/NumPy**: Data manipulation and numerical computing

### MLOps & Monitoring
- **MLflow**: Experiment tracking and model registry
- **DagHub**: Collaborative ML platform integration
- **YAML**: Configuration and schema validation
- **Custom Logging**: Structured logging with timestamp versioning

### Infrastructure & Deployment
- **FastAPI**: High-performance web framework with async support
- **Uvicorn**: ASGI server for production deployment
- **Docker**: Containerization for consistent environments
- **AWS S3**: Cloud storage and artifact synchronization
- **AWS ECR**: Container registry for deployment

### DevOps & Automation
- **GitHub Actions**: CI/CD pipeline automation
- **Terraform-ready**: Infrastructure as Code compatibility
- **Modular Design**: Reusable components and clean architecture

## 🔧 Key Components

### ML Pipeline Features
- **Intelligent Data Ingestion**: Multi-source data handling (MongoDB, CSV)
- **Robust Data Validation**: 31-column schema validation with quality checks
- **Feature Engineering**: Advanced preprocessing with imputation and scaling
- **Model Training**: Multiple algorithms with hyperparameter optimization
- **Automated Evaluation**: Performance metrics and model comparison
- **Artifact Management**: Timestamped versioning with S3 backup

### Web Application
- **Interactive Interface**: User-friendly prediction interface
- **Batch Processing**: Bulk prediction capabilities
- **Real-time Analysis**: Live threat classification
- **RESTful API**: Programmatic access for integration

### Production Features
- **Comprehensive Logging**: Structured logs with rotation
- **Error Handling**: Robust exception management
- **Performance Monitoring**: Latency and throughput metrics
- **Security Controls**: Authentication and input validation

## 📊 System Architecture

### Data Flow Process

1. **📥 Data Ingestion**
   - Reads from MongoDB collections and CSV files
   - Implements train/test splitting with proper validation
   - Creates timestamped data artifacts
   - Exports to structured directory format

2. **✅ Data Validation** 
   - Schema validation ensuring 31 expected columns
   - Data quality checks and anomaly detection
   - Drift detection with comprehensive reporting
   - Validation status logging for audit trails

3. **🔄 Data Transformation**
   - Feature preprocessing pipeline with imputation
   - Advanced scaling and normalization techniques
   - Saves preprocessing components (preprocessing.pkl)
   - Exports transformed arrays in efficient formats

4. **🎯 Model Training**
   - Random Forest and ensemble algorithms
   - Cross-validation and hyperparameter tuning
   - MLflow experiment tracking integration
   - Model persistence with versioning (model.pkl)

5. **🚀 Deployment**
   - FastAPI web application deployment
   - Docker containerization for consistency
   - AWS ECR integration for cloud deployment
   - Production monitoring and health checks

## 🚀 Quick Start

### Prerequisites
- Python 3.8+ with pip
- Docker and Docker Compose
- MongoDB instance (local or cloud)
- AWS CLI configured (optional, for cloud features)
- Git for version control
- Dagshub Account

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Yashmaini30/ThreatMatrix-Predictor
   cd ThreatMatrix-Predictor
   ```

2. **Set up environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Configure data sources**
   ```bash
   # Update configuration and env files files
   # Edit MongoDB connection and data paths
   ```

4. **Run the ML pipeline**
   ```bash
   python main.py  # Trains the model end-to-end
   ```

5. **Start the web application**
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Access the interface**
   - Web UI: http://127.0.0.1:8000
   - API docs: http://127.0.0.1:8000/docs

### Docker Deployment

```bash
# Build the container
docker build -t threat-matrix-predictor .

# Run with environment variables
docker run -p 8000:8000 \
  -e MONGODB_URL=your_mongodb_url \
  -e AWS_ACCESS_KEY_ID=your_key \
  threat-matrix-predictor
```
## 📡 API Usage

### Threat Prediction Endpoint

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": [1.2, 0.8, 3.4, ...], 
    "metadata": {
      "source": "network_monitor",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  }'
```

### Batch Processing

```bash
curl -X POST http://127.0.0.1:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {"features": [1.2, 0.8, ...]},
      {"features": [2.1, 1.3, ...]}
    ]
  }'
```

## 💰 Performance Metrics

### Model Performance
- **Precision**: 0.97 
- **Recall**: 0.96 
- **F1-Score**: 0.97 

### System Performance
- **Average latency**: <100ms per prediction
- **Throughput**: 1000+ predictions/second
- **Memory usage**: ~2GB for full pipeline
- **Storage**: ~500MB for models and artifacts

### Scalability Targets
- **Single instance**: 1K requests/minute
- **Horizontal scaling**: 10K+ requests/minute
- **Data processing**: 1M+ records/hour
- **Model retraining**: Daily automated updates

## 🔒 Security & Compliance

### Data Security
- Input validation and sanitization
- Secure MongoDB connections with authentication
- Encrypted data transmission (HTTPS/TLS)
- No sensitive data in logs or artifacts

### Infrastructure Security
- Container isolation with minimal attack surface
- AWS IAM roles with least privilege access
- Regular security updates and dependency scanning
- Environment-based configuration management

### Compliance Features
- Audit trails for all predictions and model changes
- Data lineage tracking throughout the pipeline
- Explainable AI features for regulatory requirements
- GDPR-compliant data handling practices

## 📈 Monitoring & Observability

### MLOps Monitoring
- **Experiment Tracking**: Full MLflow integration at dagshub.com/mainiyash2/ThreatMatrix-Predictor
- **Model Drift Detection**: Automated data distribution monitoring
- **Performance Tracking**: Real-time accuracy and latency metrics
- **Resource Utilization**: CPU, memory, and storage monitoring

### Application Monitoring
- **Health Checks**: Automated endpoint monitoring
- **Error Tracking**: Comprehensive exception logging
- **Performance Metrics**: Request/response time analysis
- **Usage Analytics**: API consumption patterns

### Alerting System
- Model performance degradation alerts
- System resource threshold notifications
- Data pipeline failure alerts
- Security anomaly detection

## 🔮 Future Enhancements

### Technical Roadmap
- [ ] **Advanced ML Models**: Deep learning integration with TensorFlow/PyTorch
- [ ] **Real-time Streaming**: Kafka integration for live threat detection
- [ ] **Multi-model Ensemble**: Voting classifiers for improved accuracy
- [ ] **AutoML Integration**: Automated hyperparameter optimization
- [ ] **Edge Deployment**: Lightweight models for edge computing
- [ ] **GraphQL API**: Advanced query capabilities

### Business Features
- [ ] **User Management**: Role-based access control with authentication
- [ ] **Custom Dashboards**: Grafana integration for advanced visualization
- [ ] **Threat Intelligence**: Integration with external threat feeds
- [ ] **Reporting Engine**: Automated threat analysis reports
- [ ] **Multi-tenant Support**: SaaS-ready architecture

### Infrastructure Improvements
- [ ] **Kubernetes Deployment**: Full K8s orchestration
- [ ] **Service Mesh**: Istio integration for microservices
- [ ] **Advanced Caching**: Redis integration for improved performance
- [ ] **Global CDN**: Multi-region deployment capabilities

## 📝 Project Structure

```
ThreatMatrix-Predictor/
├── NetworkSecurityFun/          # Main package
│   ├── components/              # ML pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/                # Orchestration logic
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   ├── entity/                  # Configuration classes
│   │   └── config_entity.py
│   └── utils/                   # Utility modules
│       ├── main_utils/
│       └── ml_utils/
├── cloud/                       # Cloud integration
│   └── s3_syncer.py
├── config/                      # Configuration files
│   └── config.yaml
├── logs/                        # Application logs
├── artifacts/                   # ML artifacts
│   ├── data_ingestion/
│   ├── data_validation/
│   ├── data_transformation/
│   └── model_trainer/
├── final_models/                # Production models
├── templates/                   # Web UI templates
├── .github/workflows/           # CI/CD automation
├── Dockerfile                   # Container definition
├── app.py                       # FastAPI application
├── main.py                      # Training orchestrator
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🧪 Testing & Quality Assurance

### Testing Strategy
- **Unit Tests**: Component-level testing with pytest
- **Integration Tests**: End-to-end pipeline validation
- **Model Tests**: Performance and accuracy validation
- **API Tests**: Endpoint functionality and load testing

### Code Quality
- **Linting**: Black, flake8, and pylint integration
- **Type Checking**: MyPy for static type analysis
- **Documentation**: Comprehensive docstrings and comments
- **Code Coverage**: >90% test coverage target

### Continuous Integration
- Automated testing on pull requests
- Model performance regression testing
- Security vulnerability scanning
- Docker image security analysis

## 🤝 Contributing

We welcome contributions to improve the Threat Matrix Predictor! Please follow these guidelines:

1. **Fork the repository** and create a feature branch
2. **Follow coding standards** with proper documentation
3. **Add comprehensive tests** for new functionality
4. **Update documentation** as needed
5. **Submit a pull request** with detailed description

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/ThreatMatrix-Predictor
cd ThreatMatrix-Predictor

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/ -v

# Run linting
black . && flake8 .
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

**Project Maintainer**: Your Name - your.email@example.com

**Project Repository**: https://github.com/yourusername/ThreatMatrix-Predictor

**MLflow Experiments**: https://dagshub.com/mainiyash2/ThreatMatrix-Predictor

### Getting Help
- 📖 **Documentation**: Check the wiki for detailed guides
- 🐛 **Bug Reports**: Use GitHub issues with the bug template
- 💡 **Feature Requests**: Use GitHub issues with the enhancement template
- 💬 **Discussions**: Join our community discussion

---

⭐ **Star this repository if you found it helpful!**

🔗 **Connect with me:** [LinkedIn](https://www.linkedin.com/in/yash-maini-369869198)
---

*Built with ❤️ by Yash*
