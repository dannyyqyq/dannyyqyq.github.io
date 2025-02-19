---
layout: post
title: Student Performance Classification End-to-End Deployment Project
image: "/posts/student_performance.webp"
tags: [Python, Machine Learning, Student Performance, Data Science, Flask, Docker, AWS, Azure]
github_repo: "[dannyyqyq/Student_performance](https://github.com/dannyyqyq/Student_performance)"
---

# 🎓Student Performance Classification End-to-End Deployment Project
For more details, check out the [project repository on GitHub](https://github.com/dannyyqyq/Student_performance).
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
student_performance/
├── .ebextensions/
│   └── python.config
├── artifacts/
│   ├── model.pkl
│   └── preprocessing.pkl
├── src/
│   ├── component/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   └── model_trainer.py
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── prediction_pipeline.py
│   ├── __init__.py
│   ├── exception.py
│   ├── logger.py
│   └── utils.py
├── templates/
│   ├── home.html
│   └── index.html
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── README.md
├── application.py
├── requirements.txt
└── setup.py
```

## 🚀 How It Works

### 1️⃣ Data Ingestion & Transformation
- Reads CSV data, splits it into train/test sets.
- Preprocesses features by scaling and encoding categorical data.

Here's a snippet from `data_ingestion.py`:

```python
def initiate_data_ingestion(self):
    df = pd.read_csv("notebooks/data/stud.csv")
    train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)
    # ... (further processing)
    return train_set, test_set
```

### 2️⃣ Model Training
- Trains multiple regression models and evaluates them.
- Selects the best model based on the **R² score**.
A snippet from `model_trainer.py` where models are trained:

```python
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    # ... other models
}

model_report = evaluate_model(X_train, y_train, X_test, y_test, models=models)
best_model_score = max(model_report.values(), key=lambda x: x['Test r2 score'])
```

### 3️⃣ Prediction Pipeline
- Loads the trained model and preprocessor.
- Transforms new input data and makes predictions.
  
```python
def predict(self, features):
    model = load_object(file_path="artifacts/model.pkl")
    preprocessor = load_object(file_path="artifacts/preprocessing.pkl")
    data_scaled = preprocessor.transform(features)
    return model.predict(data_scaled)
```

### 4️⃣ Web Application
- Uses Flask for a simple web UI.
- Allows users to input student features and get predicted scores.
  
```python
from flask import Flask, request, render_template
from src.pipeline.prediction_pipeline import CustomData, PredictionPipeline

app = Flask(__name__)

@app.route("/predict_data", methods=["GET", "POST"])
def predict_data():
    if request.method == "POST":
        data = CustomData(
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("race_ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            writing_score=int(request.form.get("writing_score")),
            reading_score=int(request.form.get("reading_score")),
        )
        pred_df = data.get_data_as_dataframe()
        prediction_pipeline = PredictionPipeline()
        results = prediction_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])
    return render_template("home.html")
```

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
