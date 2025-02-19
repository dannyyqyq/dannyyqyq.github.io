---
layout: post
title: Student Performance End-to-End Deployment Project
image: "/posts/student_performance.webp"
tags: [Python, Machine Learning, Student Performance, Data Science, Flask, Docker, AWS, Azure]
---

# 🎓 End-to-End Student Performance Prediction Project

## 📌 Project Overview
This project aims to predict student performance based on academic and socio-economic factors, helping educators and policymakers make data-driven decisions. The project covers:

- **Data Handling**: Ingestion and preprocessing of student data.
- **Model Training**: Experimentation with multiple regression models to predict math scores.
- **Prediction Pipeline**: Application of trained models to new data for predictions.
- **Web Application**: A Flask-based web interface for users to input student data and get predictions.
- **Deployment**: Utilization of Docker for containerizing the application and hosting on Azure Web Apps.

## 🛠 Tools and Technologies Used

### 🚀 Deployment & CI/CD
- **Amazon EC2 & GitHub Actions self-hosted runners**  
  - EC2 instances were configured as GitHub Actions self-hosted runners to provide a controlled CI/CD environment.
  - These runners executed build and test jobs, ensuring compatibility with production.
- **Amazon Elastic Container Registry (ECR)**  
  - Served as the repository for Docker images. Built images were pushed to ECR, ensuring the latest version was always available for deployment.
- **Azure Container Registry (ACR)**  
  - Used alongside AWS ECR for a multi-cloud image management strategy, allowing redundancy and Azure-based deployments.
- **Azure Web App**  
  - Hosted the final application by pulling the Docker image from ACR, ensuring high availability and easy scalability.

### 📊 Machine Learning
- **Regression Models**: 
  - Linear Regression, Decision Trees, Random Forests, XGBoost
- **Evaluation Metrics**:
  - R² Score, Mean Absolute Error (MAE)

## 📂 Project Structure

```
student_performance_project/
│── src/components/
│   ├── data_ingestion.py  # Reads and splits data
│   ├── data_transformation.py  # Preprocessing (scaling, encoding)
│   ├── model_trainer.py  # Trains ML models
│
│── src/pipeline/
│   ├── prediction_pipeline.py  # Manages the prediction pipeline
│
│── main/
│   ├── setup.py  # Project setup
│   ├── application.py  # Flask app
│   ├── requirements.txt  # Python dependencies
│   ├── Dockerfile  # Container configuration
│   ├── .pre-commit-config.yaml  # Pre-commit hooks
```

## 🚀 How It Works

### 1️⃣ Data Ingestion & Transformation
- Reads CSV data, splits it into train/test sets.
- Preprocesses features by scaling and encoding categorical data.

### 2️⃣ Model Training
- Trains multiple regression models and evaluates them.
- Selects the best model based on the **R² score**.

### 3️⃣ Prediction Pipeline
- Loads the trained model and preprocessor.
- Transforms new input data and makes predictions.

### 4️⃣ Web Application
- Uses Flask for a simple web UI.
- Allows users to input student features and get predicted scores.

### 5️⃣ Deployment
- **Docker**: Application is containerized for easy deployment.
- **CI/CD**: Automated deployment using GitHub Actions and multi-cloud setup (AWS & Azure).

## 💻 Running the Project

### 🏗️ Local Development
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run Data Pipeline**:
   ```python
   from src.components.data_ingestion import DataIngestion
   from src.components.data_transformation import DataTransformation
   from src.components.model_trainer import ModelTrainer

   obj = DataIngestion()
   train_data, test_data = obj.initiate_data_ingestion()
   
   data_transformation = DataTransformation()
   train_array, test_array, _ = data_transformation.initiate_data_transformation(train_data, test_data)

   model_trainer = ModelTrainer()
   model_trainer.initiate_model_trainer(train_array, test_array)
   ```
3. **Start Flask Server**:
   ```bash
   python application.py
   ```

### 🐳 Docker Deployment
1. **Build Docker Image**:
   ```bash
   docker build -t student_performance_predictor .
   ```
2. **Run Docker Container**:
   ```bash
   docker run -p 5000:5000 student_performance_predictor
   ```

## 🔮 Next Steps / Potential Improvements
- **📈 Model Enhancement**:
  - Explore Deep Learning models (Neural Networks, Transformers).
  - Implement model versioning and automated retraining with new data.
- **⚡ Real-time Data Processing**:
  - Integrate streaming data (Kafka, AWS Kinesis) for live predictions.
- **🧪 Automated Testing**:
  - Expand CI/CD to include unit, integration, and end-to-end tests.

---
🔥 *This project helped me deepen my knowledge in cloud-native development, CI/CD automation, and containerization. I plan to build on this by integrating real-time data and improving model performance!* 🚀
