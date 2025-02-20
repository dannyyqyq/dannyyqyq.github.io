---
layout: post
title: NLP Kindle Review Classification End-to-End Deployment Project
image: "/posts/kindle_review.png"
tags: [Python, NLP, Machine Learning, Sentiment Analysis, Streamlit, Docker, Word2Vec, AvgWord2Vec, TF-IDF, BoW]
github_repo: "[dannyyqyq/NLP_kindle_classification](https://github.com/dannyyqyq/NLP_kindle_classification)"
---

# 📚 NLP Kindle Review Classification End-to-End Deployment Project
For more details, check out the [project repository on GitHub](https://github.com/dannyyqyq/NLP_kindle_classification).

## 🚀 Web Application
Experience the Kindle Review Sentiment Analysis live!  
[Live Demo: Kindle Review Sentiment Analysis](https://kindle-review-ratings-classification-avgword2vec.streamlit.app/)

## 📌 Project Overview
This project leverages Natural Language Processing (NLP) to classify Kindle book reviews into positive or negative sentiments, aiding in understanding reader feedback, achieving up to 65% accuracy using AvgWord2Vec embeddings and a Random Forest Classifier model. The project covers:

- **Data Ingestion**: Extracting and preparing review data from CSV files for analysis.
- **Data Transformation**: Preprocessing text data for model training.
- **Model Training**: Training multiple NLP models to classify review sentiments.
- **Prediction Pipeline**: Applying trained models to predict sentiments on new reviews.
- **Web Application**: A Streamlit-based interface for users to input reviews and get sentiment predictions.
- **Deployment**: Deployed on Streamlit Cloud for live access, with optional Docker containerization for local testing or custom deployment setups.
  
## 🛠 Tools and Technologies Used

### 🚀 Deployment
- **Docker**: 
  - Employed for containerization, ensuring uniform deployment across environments.
- **Streamlit**: 
  - Provides a user-friendly web interface for real-time sentiment analysis.

### 📊 Machine Learning / NLP
- **Classification Models**: 
  - Random Forest, Naive Bayes (Gaussian), with feature extraction via Bag of Words (BoW), Term Frequency-Inverse Document Frequency (TF-IDF), Word2Vec, and AvgWord2Vec.
- **Evaluation Metrics**: 
  The performance of different feature extraction techniques and classification models is summarized below:

  | Feature Extraction       | Classification Model    | Accuracy Score (%) |
  |--------------------------|-------------------------|---------------------|
  | AvgWord2Vec              | Random Forest           | 64.2%                |
  | BOW       | Naive Bayes             | 54.3%                |
  | TF-IDF | Naive Bayes | 54.8%                |

## 📂 Project Structure

```plaintext
NLP_kindle_classification/
├── artifacts/
│   ├── best_model.pkl
│   └── word2vec_model.pkl
├── data/
│   └── all_kindle_review.csv
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── data_validation.py
│   │   └── model_trainer.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── utils.py
│   ├── exception/
│   │   ├── __init__.py
│   │   └── exception.py
│   └── logger/
│       ├── __init__.py
│       └── logger.py
├── .gitignore
├── .pre-commit-config.yaml
├── Dockerfile
├── README.md
├── app.py
├── main.py
├── requirements.txt
└── setup.py
```

## 🚀 How It Works

### 1️⃣ Data Ingestion
- Reads Kindle review data from a CSV file and saves it for further processing.

Here's a snippet from `data_ingestion.py`:
```python
def initiate_data_ingestion(self):
    try:
        df = pd.read_csv(self.ingestion_config.base_data_path)
        logging.info("Read the dataset as dataframe")
        directory = os.path.dirname(self.ingestion_config.raw_data_path)
        os.makedirs(directory, exist_ok=True)
        df.to_csv(self.ingestion_config.raw_data_path, index=False)
        return df, self.ingestion_config.raw_data_path
    except Exception as e:
        raise CustomException(e, sys)
```

### 2️⃣ Data Transformation
- Preprocesses review text by cleaning and normalizing it, converting ratings to binary sentiment labels.

Here's a snippet from `data_ingestion.py`:
```python
def data_transformation(self, df):
    try:
        df = df[["reviewText", "rating"]].copy()
        df["rating"] = (df["rating"] >= 3).astype(int)  
        # Convert rating: 1 if >=3, else 0
        df["reviewText"] = df["reviewText"].astype(str).str.lower()  
        # Convert to lowercase
        df["reviewText"] = df["reviewText"].apply(
            lambda x: self.special_chars_pattern.sub("", x)  
        # Remove special characters
        )
        return df
    except Exception as e:
        raise CustomException(e, sys)
```

### 3️⃣ Feature Extraction and Model Training
- Trains models using multiple feature extraction techniques and selects the best based on accuracy.

**Bag of Words (BoW)**
- Using BOW, Converts text into a matrix of word counts, used with a Naive Bayes classifier.
A snippet from `model_trainer.py`:
```python
bow = CountVectorizer()
X_train_bow = bow.fit_transform(X_train).toarray()
X_test_bow = bow.transform(X_test).toarray()
nb_model_bow = GaussianNB().fit(X_train_bow, y_train)
y_pred_bow = nb_model_bow.predict(X_test_bow)
accuracy_score_bow = accuracy_score(y_test, y_pred_bow)
```

**Term Frequency-Inverse Document Frequency(TF-IDF)**
- Transforms text into TF-IDF features (term frequency-inverse document frequency), used with Naive Bayes
A snippet from `model_trainer.py`:
```python
tfidf = TfidfVectorizer()
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()
nb_model_tfidf = GaussianNB().fit(X_train_tfidf, y_train)
y_pred_tfidf = nb_model_tfidf.predict(X_test_tfidf)
accuracy_score_tfidf = accuracy_score(y_test, y_pred_tfidf)
```

**Word2Vec**
- Trains a Word2Vec model to generate word embeddings for the review text.
A snippet from `model_trainer.py`:
```python
def training_dataset_Word2Vec(df, vector_size=100):
    model = Word2Vec(vector_size=vector_size, window=5)
    words = [simple_preprocess(review) for review in df["reviewText"]]
    model.build_vocab(words)
    model.train(words, total_examples=len(words), epochs=model.epochs)
    return df_final, model
```

**AvgWord2Vec**
- Computes the average Word2Vec vector for each review, used with Random Forest.
A snippet from `model_trainer.py`:
```python
def avg_word2vec(model, doc):
    return np.mean(
        [model.wv[word] for word in doc if word in model.wv.index_to_key], axis=0)
X = [ModelTrainer.avg_word2vec(model, review_words) for review_words in words]
```

**Why AvgWord2Vec over Word2Vec**
- Word2Vec: Creates a 100D vector per word. For "This book is great," it might yield [0.2, 0.3, ...] (This), [0.6, 0.7, ...] (book), [0.1, 0.1, ...] (is), [0.8, 0.9, ...] (great)—a variable-length list incompatible with Random Forest, which needs one fixed-size input. For "Reading feels dull," it could produce [0.4, 0.5, ...] (Reading), [0.3, 0.4, ...] (feels), [0.5, 0.6, ...] (dull)—another variable list.
- AvgWord2Vec: Averages these into one 100D vector per review. For "This book is great," it might become [0.425, 0.5, ...]; for "Reading feels dull," it might yield [0.4, 0.5, ...]. This fixed-size vector ensures compatibility with classifiers while summarizing semantic content, despite losing word order.

### 4️⃣ Prediction Pipeline
- Loads the best model and Word2Vec model to predict sentiments. In this case, the best model uses Word2Vec model embeddings.
- Returns None to Safeguard Against Invalid Input. If vector is None, it means the review couldn’t be vectorized (e.g., all words are unknown to the model). Attempting to reshape or predict on None would cause an error, so returning None prevents crashes and signals to the caller (e.g., the Streamlit app) that no prediction is possible.
```python
def predict(self, features):
    vector = avg_word2vec(w2v_model, features)
    if vector is not None:
        vector = vector.reshape(1, -1)
        return model.predict(vector)
    return None
```

### 5️⃣ Web Application
- Uses Streamlit to allow users to input reviews and view sentiment predictions.
```python
import streamlit as st
from src.components.data_transformation import DataTransformation

st.title("Kindle Review Sentiment Analysis")
review_text = st.text_area("Enter your review text here:", "")
if st.button("Analyze"):
    if review_text:
        data_transformer = DataTransformation()
        df = pd.DataFrame([{'reviewText': review_text, 'rating': 0}])
        df_transformed = data_transformer.data_transformation(df.copy())
        df_transformed = data_transformer.lemmatizer_transformation(df_transformed)
        prediction = predict(df_transformed['reviewText'].iloc[0])
        sentiment = "Positive" if prediction[0] == 1 else "Negative"
        st.write(f"The sentiment of the review is: **{sentiment}**")
```

### 6️⃣ Deployment
- **Streamlit Cloud**: Deployed directly on Streamlit Cloud for web access. No Docker required for cloud deployment.
- **Optional Local Docker Deployment** (for testing or custom setups)- refer to docker deployment below

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
    df, _ = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_set, test_set, _, _ = data_transformation.initiate_data_transformation(df)
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_trainer(train_set, test_set)
   ```
3. **Start Streamlit Server**:
   ```bash
    streamlit run app.py
   ```

🐳 **Docker Deployment** (Optional for Local Testing)
1. **Build Docker Image**:
   ```bash
    docker build -t nlp_kindle_system .
    ```

2. **Run Docker Container**:
   ```bash
    docker run -p 8501:8501 nlp_kindle_system
   ```

## ⚠️ Challenges Faced

* **Feature Extraction:**
  - Choosing the optimal feature extraction method proved to be a significant challenge. We explored various techniques, including TF-IDF and Bag of Words, before discovering that AvgWord2Vec embeddings yielded the most effective results for our classification models. 
* **Deployment:**
  - Deploying the Streamlit application on Streamlit Cloud initially resulted in deployment failures due to unresolved environment dependencies. To effectively diagnose the issue, I containerized the application using Docker, allowing me to replicate the Streamlit Cloud environment locally. This process enabled me to pinpoint the root cause of the errors: the NLTK stopwords library was not consistently available. To address this, I modified the `app.py` script to ensure the NLTK stopwords library is downloaded every time the application runs, guaranteeing its presence in the Streamlit Cloud environment.

  
## 🔮 Next Steps / Potential Improvements
- **📈 Model Enhancement**:
- Explore transformer models like BERT, or more advance NLP language models for improved accuracy, or consider tuning windows size and ngrams.
- **⚡ Real-time Data Processing**:
  - Integrate streaming data (Kafka, AWS Kinesis) for live predictions.
- **🧪 Automated Testing**:
  - Expand CI/CD to include unit, integration, and end-to-end tests.
  
🔥 This project showcases my journey in applying NLP to classify Kindle reviews, enhancing my skills in sentiment analysis and web deployment. 🚀