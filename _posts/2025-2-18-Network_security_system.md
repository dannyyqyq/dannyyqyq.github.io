---
layout: post
title: Network Security System for Phishing Data End to End Deployment Project
image: "/posts/network.jpg"
tags: [Python, Machine Learning, Network Security, MongoDB, MLflow, FastAPI, AWS, Docker]
github_repo: "[dannyyqyq/Student_performance](https://github.com/dannyyqyq/network_security)"
---

# 🔒 Network Security System for Phishing Data End-to-End Deployment Project
For more details, check out the [project repository on GitHub](https://github.com/dannyyqyq/network_security).

## 📌 Project Overview

This project develops a system to detect phishing attempts by analyzing network data. It combines machine learning techniques with secure data storage and processing. This project covers:

- **Data Handling**: Efficient management of network data using MongoDB for secure storage and retrieval.
- **Data Ingestion**: Extracting data from various sources, converting it into a usable format, and preparing it for analysis.
- **Model Training**: Application of machine learning techniques like Random Forest, Decision Trees, and Gradient Boosting to train models for identifying phishing patterns.
- **Prediction Pipeline**: Utilizing trained models to make real-time predictions on new data.
- **Web Application**: Development of a FastAPI-based web application that serves as an interface for data input and prediction output.
- **Deployment**: Containerizing the application with Docker and employing CI/CD practices for seamless deployment across different environments, ensuring scalability and reliability.
  
## 🛠 Tools and Technologies

- **Python**: Core language for development.
- **MongoDB**: For data storage, particularly using MongoDB Atlas for cloud-based solutions.
- **Machine Learning**: Using Scikit-learn for model training, alongside MLflow for experiment tracking.
- **FastAPI**: For building a fast, modern, Python-based web API.
- **Docker**: For containerization and easy deployment.
- **CI/CD**: GitHub Actions for continuous integration and deployment.
- **Others**: `pymongo` for MongoDB interactions, `dill` for serialization, `pandas` for data manipulation, `certifi` for SSL certificate handling, AWS S3 for cloud storage.

### 🚀 Deployment & CI/CD
- **GitHub Actions**: Utilized for an automated CI/CD pipeline, ensuring code quality and consistency from development to production.
- **Docker**: Employed for containerization, which supports uniform deployment across different environments.
- **AWS S3**: Acts as storage for model artifacts, data backups, and logs, facilitating data preservation and recovery.
- **MongoDB Atlas**: Provides a scalable and secure database solution for data storage and management in the cloud.
  
## 📊 Key Features

- **Data Extraction and Storage**: Converts CSV to JSON, stores in MongoDB, and syncs with AWS S3.
- **Data Pipeline**: Comprehensive pipeline for data ingestion, validation, transformation, and model training.
- **Machine Learning**: Trains models with various classifiers to identify phishing patterns.
- **API**: Provides endpoints for real-time phishing detection:
  - **Training Pipeline**: Starts the training pipeline via an API call.
  - **Prediction**: Accepts CSV files for on-the-fly prediction and returns results in HTML.
- **Model and Data Sync**: Syncs local artifacts and models to S3 for cloud storage.

## 🚀 Project Structure

```plaintext
network_security/
├── .github/
│   └── workflows/
│       └── main.yml
├── config/
│   └── database_config.py
├── data_schema/
│   └── schema.yaml
├── final_model/
│   ├── model.pkl
│   └── preprocessor.pkl
├── mlruns/
├── network_security/
│   ├── cloud/
│   │   ├── __init__.py
│   │   └── s3_syncer.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   └── model_trainer.py
│   ├── constants/
│   │   └── training_pipeline/
│   │       ├── __init__.py
│   │       └── __init__.py
│   ├── entity/
│   │   ├── __init__.py
│   │   ├── artifact_entity.py
│   │   └── config_entity.py
│   ├── exception/
│   │   ├── __init__.py
│   │   └── exception.py
│   ├── logging/
│   │   ├── __init__.py
│   │   └── logger.py
│   ├── notebooks/
│   │   └── placeholder.ipynb
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── training_pipeline.py
│   └── utils/
│       ├── main_utils/
│       │   └── __init__.py
│       └── ml_utils/
│           ├── metric/
│           │   ├── __init__.py
│           │   └── classification_metric.py
│           ├── model/
│           │   ├── __init__.py
│           │   └── estimator.py
│           └── __init__.py
├── templates/
│   └── table.html
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── README.md
├── app.py
├── main.py
├── push_data.py
├── requirements.txt
└── setup.py
```

## 🚀 How It Works

### 1️⃣ Data Ingestion & Transformation
- Reads data from MongoDB, converts it to CSV, and splits it into train/test sets.
- Preprocesses data, handling missing values with KNN Imputer.

Here's a snippet from `data_ingestion.py`:

```python
def export_collection_as_dataframe(self):
    database_name = self.data_ingestion_config.database_name
    collection_name = self.data_ingestion_config.collection_name
    self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
    collection = self.mongo_client[database_name][collection_name]
    df = pd.DataFrame(list(collection.find()))
    if "_id" in df.columns.to_list():
        df = df.drop(columns=["_id"], axis=1)
    df.replace({"na": np.nan}, inplace=True)
    return df
```

### 2️⃣ Model Training
- Trains multiple classification models with hyperparameter tuning.
- Selects the best model based on performance metrics like F1 score.
A snippet from `model_trainer.py` where models are trained:

```python
models = {
    "Random Forest": RandomForestClassifier(verbose=1),
    "Decision Tree": DecisionTreeClassifier(),
    # ... other models
}

params = {
    "Decision Tree": {'criterion':['gini', 'entropy', 'log_loss']},
    "Random Forest": {'n_estimators': [8,16,32,128,256]},
    # ... other params
}
model_report = evaluate_models(X_train, y_train, X_test, y_test, models=models, param=params)
best_model_score = max(sorted(model_report.values()))
```

### 3️⃣ Prediction Pipeline
- Loads the trained model and preprocessor for predictions.
- Transforms new input data and makes predictions.

```python
def predict(self, x):
    x_transform = self.preprocessor.transform(x)
    y_hat = self.model.predict(x_transform)
    return y_hat
```

### 4️⃣ Web Application
- Uses FastAPI for a fast, modern API.
- Provides endpoints for training models and making predictions.
  
```python
from fastapi import FastAPI, File, UploadFile, Request

app = FastAPI()

@app.post("/predict")
async def predict_route(request: Request, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)
    preprocessor = load_object("final_model/preprocessor.pkl")
    model = load_object("final_model/model.pkl")
    network_model = NetworkModel(preprocessor=preprocessor, model=model)
    y_pred = network_model.predict(df)
    df["predicted_column"] = y_pred
    df.to_csv("prediction_output/output.csv")
    table_html = df.to_html(classes="table table-striped")
    return templates.TemplateResponse("table.html", {"request": request, "table": table_html})
```

### 5️⃣ Deployment
- **Docker**: Application is containerized for easy deployment.
- **Cloud Sync**: Syncs artifacts and models to AWS S3.

## 💻 Running the Project

### 🏗️ Local Development
1. **Install Dependencies**:
   ```bash
    pip install -r requirements.txt
   ```
2. **Run Data Pipeline**:
   ```python
    from network_security.pipeline.training_pipeline import TrainingPipeline
    pipeline = TrainingPipeline()
    pipeline.run_pipeline()
   ```
3. **Start FastAPI Server**:
   ```bash
   uvicorn app:app --reload
   ```

### 🐳 Docker Deployment
1. **Build Docker Image**:
   ```bash
   docker build -t network_security_system .
   ```
2. **Run Docker Container**:
   ```bash
   docker run -p 8080:8080 network_security_system
   ```

## ⚠️ Challenges Faced
- **AWS Environment Setup via IAM**:
  - Configuring the AWS environment for the network security system proved challenging due to the need to set up Identity and Access Management (IAM) roles and policies. 
- **Ease of Azure Web App Setup with Container Registry**:
  - Deploying the network security system on Azure Web App (Azure App Service) with a containerized approach was straightforward. I needed to configure an Azure Container Registry, push the Docker image to the registry, deploy the containerized application to Azure Web App via GitHub Actions, and link the repository to GitHub for CI/CD actions.
- **Managing Secrets in GitHub Self-Hosted Actions**:
  - Integrating secret manager keys and passwords into GitHub self-hosted actions presented difficulties. Ensuring secure storage and retrieval of sensitive credentials (e.g., API keys, database passwords) required configuring GitHub Secrets and Actions workflows correctly. 

## 🔮 Next Steps / Potential Improvements
- **📈 Model Enhancement**:
  - Explore Recurrent Neural Networks (RNNs) or Transformers for analyzing the sequential nature of network traffic data
  - Implement model versioning and automated retraining with new data patterns.
- **⚡ Real-time Data Processing**:
  - Integrate streaming data solutions like Kafka/Redpanda for continuous data analysis.
- **🧪 Automated Testing**:
  - Expand CI/CD to include more comprehensive testing strategies.

---
🔥 *This project showcases my journey in integrating MongoDB for data management, applying machine learning to enhance network security, and focusing on practical phishing detection!* 🚀
