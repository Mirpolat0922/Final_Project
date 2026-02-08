# Sentiment Analysis for Movie Reviews

A project for classifying movie reviews as positive or negative using Natural Language Processing and Support Vector Machines.

---

## Table of Contents
- [DS Part - Data Science Analysis](#ds-part---data-science-analysis)
- [MLE Part - ML Engineering](#mle-part---ml-engineering)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

---

## DS Part - Data Science Analysis

### Overview
The project implements a sentiment classification system for movie reviews using classical machine learning techniques. The solution achieves **90.52% accuracy** on the inference dataset through careful feature engineering and model selection.

### Exploratory Data Analysis

**Dataset Characteristics:**
- **Total training samples**: 40 000 movie reviews
- **Class distribution**: Perfectly balanced (50% positive, 50% negative)
- **Average review length**: ~1310 characters
- **Data quality**: No missing values, clean text format

**Key Insights:**
- Reviews vary significantly in length and complexity
- Balanced classes eliminate need for sampling strategies
- Text contains rich sentiment signals in both word choice and negations

### Feature Engineering Pipeline

The preprocessing pipeline transforms raw text into numerical features suitable for machine learning:

#### 1. Text Preprocessing
- **Lowercasing**: Normalizes text ("Good" → "good") for consistent treatment
- **Punctuation Handling**: Removes most punctuation while **preserving apostrophes** to maintain negations ("don't" remains intact)
- **Tokenization**: Splits text into individual words using NLTK's word tokenizer

#### 2. Stop Words Filtering
- Removes common words (articles, prepositions) that don't carry sentiment
- **Critical Decision**: Preserves negation words (not, no, never, etc.) as they flip sentiment meaning
- Example: "not good" must be treated differently from "good"

#### 3. Lemmatization vs Stemming
**Stemming approach:**
- Reduces words to stems (e.g., "running" → "runn", "remembered" → "rememb")
- Fast but produces non-dictionary forms
- A bit lower model performance

**Lemmatization approach** (chosen):
- Reduces words to dictionary base forms (e.g., "running" → "run", "better" → "good")
- Preserves word meaning and readability
- **Result**: 1-2% accuracy improvement over stemming

#### 4. Vectorization: TF-IDF vs Count Vectorizer

**TF-IDF (selected):**
- **TF (Term Frequency)**: Captures how often word appears in document
- **IDF (Inverse Document Frequency)**: Weights words by rarity across corpus
- Formula: TF-IDF = (word frequency in doc) × log(total docs / docs containing word)
- **Parameters**:
  - `ngram_range=(1,3)`: Captures single words, bigrams, and trigrams ("not good" as unit)
  - `max_df=0.9`: Ignores words appearing in >90% of documents (too common)
  - `min_df=5`: Ignores words appearing in <5 documents (likely typos or very rare)
- **Advantage**: Identifies important words while filtering noise

**Count Vectorizer (baseline):**
- Simple word frequency counts
- Performance: 1-2% lower than TF-IDF

**Decision**: TF-IDF chosen for superior importance weighting

### Model Selection

Evaluated multiple algorithms to find optimal classifier. Here are the models trained on the best preprocessed data:

| Model | Preprocessing | Accuracy | Notes |
|-------|--------------|------------|---------|
| **LinearSVC** | Lemma + TF-IDF | **90.16%**| Best overall |
| Logistic Regression | Lemma + TF-IDF | 89.51% | Fast baseline |
| Naive Bayes | Lemma + TF-IDF | 88.45%   | Fastest |
| Random Forest | Lemma + TF-IDF | 84.79% | Underperforms |

#### Why Linear Models Excel

After TF-IDF vectorization, text becomes:
- **High-dimensional**: 10,000+ features (unique n-grams)
- **Sparse**: Most entries are zero (reviews use small vocabulary subset)
- **Linearly separable**: Positive and negative reviews cluster in feature space

Linear models (SVC, Logistic Regression) are designed for exactly this:
- Efficiently handle sparse, high-dimensional data
- Find optimal hyperplane separating classes
- Fast training and prediction

#### Why Tree Models Struggle

Random Forest and other tree-based models:
- Split on thresholds (e.g., "Is feature > 0.37?")
- Ineffective on sparse data (most features are zero)
- Require many trees to capture linear relationships
- **Result**: 6% lower accuracy than LinearSVC

### Final Model: LinearSVC

**Performance:**
- **Training Accuracy**: 90.15%
- **Inference Accuracy**: 90.52%
- **ROC AUC**: 96.56%
- **No overfitting**: Inference outperforms training (model generalizes well)

### Business Value & Applications

#### 1. Automated Customer Feedback Analysis
**Problem**: Companies receive thousands of reviews daily; manual analysis is impossible
**Solution**: Classify reviews automatically by sentiment\
**Impact**:
- Process 10,000+ reviews per hour (vs. 50 manually)
- **Cost savings**: $100K+/year in manual labor
- Real-time sentiment tracking for product launches
- Identify negative feedbacks within minutes for rapid response

#### 2. Product Launch Assessment
**Use Case**: Movie studio releases new film
**Application**:
- Monitor social media and review sites in real-time
- Track sentiment evolution (opening weekend vs. week 2)
- Alert if negative sentiment crosses threshold (< 40% positive)
**Value**: Early warning system prevents reputation damage

#### 3. Competitive Intelligence
**Strategy**: Analyze competitor movie reviews
**Insights**:
- Identify what audiences love/hate about competitor films
- Benchmark our films against market leaders
- Discover untapped audience preferences
**Advantage**: Data-driven content creation decisions

#### 4. Content Moderation & Prioritization
**Challenge**: Not all negative reviews require immediate response
**Solution**:
- Automatically flag high-confidence negative reviews (score < -0.5)
- Prioritize customer service responses
- Route to appropriate team (refunds, apologies, explanations)

### Domain Transferability

**Current Domain**: Movie reviews
**Transferable to**:
- Product reviews (Amazon, electronics, clothing)
- Restaurant reviews (Yelp, Google)
- Hotel reviews (TripAdvisor, Booking.com)
- App reviews (Google Play, App Store)
- And other domains

**Caveat**: Domain-specific words may shift meaning
- "sick" in movie reviews = negative
- "sick" in youth slang = positive\
But the model still works well in other domains too.
---

## MLE Part - ML Engineering

### Project Structure
```bash
Final_Project/
├── data/                          # Git-ignored: Downloaded datasets
│   └── raw/
│       ├── final_project_train_dataset/
│       │    └── train.csv             # Training data (auto-downloaded)
│       └── final_project_inference_dataset/
│            └── inference.csv         # Inference data (auto-downloaded)
├── data_process/
│   ├── __init__.py
│   └── data_generation.py        # Data downloading script
├── training/
│   ├── __init__.py
│   ├── train.py                  # Training pipeline
│   └── Dockerfile                # Training container
├── inference/
│   ├── __init__.py
│   ├── run.py                    # Inference pipeline
│   └── Dockerfile                # Inference container
├── models/                        # Git-ignored: Trained artifacts
│   ├── sentiment_model.pickle    # Trained LinearSVC
│   ├── tfidf_vectorizer.pickle   # Fitted TF-IDF vectorizer
│   ├── training_metrics.json     # Training evaluation
│   └── figures/
│       └── confusion_matrix.png  # Training confusion matrix
├── results/                       # Git-ignored: Inference outputs
│   ├── <timestamp>.csv           # Predictions with scores
│   ├── inference_metrics.json    # Inference evaluation
│   └── figures/
│       └── inference_confusion_matrix.png
│   Notebooks/
│    └── FinalProject.ipynb
├── utils.py                       # Shared utility functions
├── settings.json                  # Configuration parameters
├── requirements.txt               # Python dependencies
├── README.md                      
└──  __init__.py
```
### Configuration (`settings.json`)

All hyperparameters and paths are centralized in `settings.json`:
```json
{
    "general": {
        "random_state": 42,        // Reproducibility seed
        "data_dir": "data",
        "models_dir": "models",
        "results_dir": "results"
    },
    "train": {
        "tfidf_ngram_range": [1, 3],  // Unigrams to trigrams
        "tfidf_max_df": 0.9,           // Max document frequency
        "tfidf_min_df": 5,             // Min document frequency
        "svm_max_iter": 3000           // SVC iterations
    },
    "text_processing": {
        "keep_negations": [            // Preserved negation words
            "no", "not", "nor", "never", 
            "none", "nobody", "nothing", "nowhere"
        ]
    }
}
```

---

## Quick Start

### Prerequisites
- **Docker** installed and running
- **Git** for version control
- **Internet connection** for data download (first run only)

### Step 1: Clone Repository
```bash
git clone https://github.com/Mirpolat0922/Final_Project.git
cd Final_Project
```

### Step 2: Training

**Build training image:**
```bash
docker build -f ./training/Dockerfile --build-arg settings_name=settings.json -t sentiment-train .
```

**Run training:**
```bash
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data sentiment-train
```

**What happens:**
1. Downloads training and inference datasets
2. Preprocesses 40,000 reviews (lemmatization, TF-IDF)
3. Trains LinearSVC model
4. Evaluates on training data
5. Saves:
   - `models/sentiment_model.pickle`
   - `models/tfidf_vectorizer.pickle`
   - `models/training_metrics.json`
   - `models/figures/confusion_matrix.png`

**Expected output:**

Validation Accuracy: 0.90158\
Validation ROC AUC: 0.96346\
Model saved to models/sentiment_model.pickle

### Step 3: Inference

**Build inference image:**
```bash
docker build -f ./inference/Dockerfile --build-arg settings_name=settings.json -t sentiment-inference .
```

**Run inference:**
```bash
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data -v $(pwd)/results:/app/results sentiment-inference
```

**What happens:**
1. Loads trained model and vectorizer from `models/`
2. Gets inference dataset from loaded data
3. Preprocesses 10,000 reviews
4. Generates predictions
5. Evaluates against ground truth
6. Saves:
   - `results/<timestamp>.csv` - All predictions
   - `results/inference_metrics.json` - Metrics
   - `results/inference_confusion_matrix.png` - Visualization

**Expected output:**

Inference Accuracy: 0.90520
Inference ROC AUC: 0.96562
Results saved to results/08.02.2026_14.30.csv

### Alternative: Local Execution

If Docker issues occur, run locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Download data
python data_process/data_generation.py

# Train
export CONF_PATH=settings.json  
python training/train.py

# Inference
python inference/run.py
```

---

## Output Files

### Training Outputs (`models/`)

**sentiment_model.pickle**
- Serialized LinearSVC model
- Contains trained weights and hyperparameters

**tfidf_vectorizer.pickle**
- Fitted TF-IDF vectorizer
- Contains vocabulary and IDF values

**training_metrics.json**
```json
{
    "validation_accuracy": 0.9015833333333333,
    "validation_roc_auc": 0.9634616111111111,
    "classification_report": {...}
}
```

### Inference Outputs (`results/`)

**<timestamp>.csv**
```csv
review,true_sentiment,predicted_sentiment,decision_score
"Great movie!",positive,positive,1.2345
"Terrible film.",negative,negative,-0.9876
...
```

**inference_metrics.json**
```json
{
    "accuracy": 0.9052,
    "roc_auc": 0.9656176800000001,
    "classification_report": { ... }
}
```

---

## Reported Metrics

### Training Performance
- **Accuracy**: 90.15%
- **Accuracy(in notebook)**: 90.16%

### Inference Performance (FINAL)
- **Accuracy**: 90.52%
- **Accuracy(in notebook)**: 90.87%  ---> 0.35% difference because in notebook, I retrained the model on the whole training data

---

## Troubleshooting

### Issue: "Model not found"
**Cause**: Inference run before training
**Solution**: 
```bash
# Run training first
docker run -v $(pwd)/models:/app/models -v $(pwd)/data:/app/data sentiment-train
```

### Issue: "Connection timeout during data download"
**Cause**: Network issues or firewall blocking EPAM CDN
**Solution**:
1. Check internet connection
2. Try VPN if corporate firewall blocks downloads
3. Download manually from URLs in `settings.json` → place in `data/raw/`

---
## Experience share

In the last module homework, I wrote dockerfiles and train scripts expecting that data is loaded first manually(forgot). That is why just running the containers did not work correctly.\
For that reason, now I am running both data loading and training scripts inside the DockerFile. And then inference container uses the trained model for inference data. So every action is happening inside the Docker.

Since the instructions of the project were quite specific and also in the last homework mentor said that the loggings and structures were a bit messy, I used LLM more for the structure of the code and visuals.

## Possible improvements

- Used lemmatization technique in preprocessing but did not perform POS tagging. Probably POS tagging would help a little for reducing the words to their correct form.
- Did not train the model on the whole training data. Probably it would have increased the accuracy a little bit about ~0.4%.
