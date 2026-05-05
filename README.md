# Fake News Detection

A machine learning project to detect fake news using TF-IDF vectorization and classification algorithms. This project implements multiple classifiers to identify whether news articles are genuine or fake based on their content.

## 📋 Project Overview

This project analyzes a dataset of news articles with the following characteristics:
- **Dataset Size**: 20,800 articles
- **Features**: ID, title, author, text content, and label (0 = real, 1 = fake)
- **Preprocessing**: Text cleaning, lemmatization, and stop word removal
- **Models Implemented**: Passive Aggressive Classifier and MLP Classifier

## 📊 Dataset

The dataset contains:
- `id`: Article identifier
- `title`: Article headline
- `author`: Author name
- `text`: Full article text
- `label`: Classification label (0 for real news, 1 for fake news)

### Data Statistics
- Total samples: 20,800
- Missing values handled: Title (558), Author (1,957), Text (39)
- Class distribution: Balanced between real and fake news

## 🔧 Data Preprocessing

1. **Data Cleaning**
   - Removed unnecessary ID column
   - Filled missing values with empty strings
   - Combined author, title, and text into single 'content' field

2. **Text Processing**
   - Converted text to lowercase
   - Removed special characters and punctuation
   - Removed English stopwords
   - Applied lemmatization using TextBlob

3. **Feature Extraction**
   - TF-IDF Vectorizer with max 5,000 features
   - Token pattern: `\w{1,}`

## 🤖 Models

### 1. Passive Aggressive Classifier
- **Accuracy**: 96%
- **Precision**: 0.96
- **Recall**: 0.96
- **F1-Score**: 0.96

**Confusion Matrix:**
```
[[2975  141]
 [ 124 3000]]
```

### 2. MLP Classifier (Neural Network)
- **Architecture**: 3 hidden layers (256, 64, 16 neurons)
- **Activation**: ReLU
- **Optimizer**: Adam
- **Accuracy**: 57% (Note: Training was interrupted, leading to suboptimal performance)

**Confusion Matrix:**
```
[[3109    7]
 [2682  442]]
```

## 📈 Results

The **Passive Aggressive Classifier** performed significantly better with balanced precision and recall, making it the recommended model for this task.

| Metric | Passive Aggressive | MLP |
|--------|-------------------|-----|
| Precision (Class 0) | 0.96 | 0.54 |
| Recall (Class 0) | 0.95 | 1.00 |
| Precision (Class 1) | 0.96 | 0.98 |
| Recall (Class 1) | 0.96 | 0.14 |
| Overall Accuracy | 96% | 57% |

## 📦 Dependencies

```
pandas
numpy
matplotlib
seaborn
nltk
textblob
scikit-learn
```

### Installation

```bash
pip install pandas numpy matplotlib seaborn nltk textblob scikit-learn
```

## 💾 Model Serialization

The trained MLP classifier is saved as `fakenews1.pkl` using pickle for future predictions:

```python
import pickle
model = pickle.load(open("fakenews1.pkl", "rb"))
predictions = model.predict(new_data)
```

## 🚀 Usage

1. **Load Data**
   ```python
   data = pd.read_csv("fake_news.csv")
   ```

2. **Preprocess Text**
   - Apply lowercasing, special character removal, stopword removal, and lemmatization

3. **Feature Engineering**
   - Use TfidfVectorizer to convert text to numerical features

4. **Train Model**
   ```python
   clf = PassiveAggressiveClassifier()
   clf.fit(X_train_tfidf, y_train)
   predictions = clf.predict(X_test_tfidf)
   ```

5. **Evaluate**
   ```python
   from sklearn import metrics
   print(metrics.classification_report(y_test, predictions))
   ```

## 📝 Dataset Split

- **Training Set**: 70% (14,560 samples)
- **Testing Set**: 30% (6,240 samples)
- **Stratification**: Used to maintain class balance

## ⚠️ Notes

- The MLP classifier training was interrupted, resulting in suboptimal performance. Consider re-training with more epochs or different hyperparameters for better results.
- The Passive Aggressive Classifier is the recommended model due to its superior performance (96% accuracy).
- The TF-IDF vectorizer was fitted on the entire dataset and used for both train and test transformations.

## 🔮 Future Improvements

- Experiment with different classifiers (Random Forest, XGBoost, SVM)
- Implement cross-validation for better model evaluation
- Tune hyperparameters using GridSearchCV
- Use pre-trained embeddings (Word2Vec, GloVe, or BERT)
- Implement ensemble methods
- Add data augmentation techniques

## 📄 Project Files

- `Fake News Detection (12-02-2025 - 13-02-2025).ipynb` - Main Jupyter notebook with full implementation
- `fakenews1.pkl` - Serialized MLP classifier model
- `fake_news.csv` - Dataset file (not included in repo)

## 👤 Author

VKR2004

## 📅 Project Date

February 12-13, 2025

---

**Disclaimer**: This model is trained for educational purposes. Results should be validated on domain-specific data before deployment in production environments.
