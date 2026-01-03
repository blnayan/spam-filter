# Spam Filter Project

## Overview
This project detects spam emails using machine learning algorithms ranging from simple probabilistic models to state-of-the-art deep learning techniques. It utilizes the Enron Spam Dataset to train and evaluate models for spam classification.

## Models and Performance
Three distinct models were implemented and evaluated on the test set.

| Model | Test Accuracy | Spam Precision | Spam Recall | Spam F1-Score | Ham F1-Score |
|-------|---------------|----------------|-------------|---------------|--------------|
| **Naive Bayes** | 95.17% | 0.9597 | 0.9470 | 0.9533 | 0.9500 |
| **DNN (TensorFlow)** | 97.38% | 0.9668 | 0.9834 | 0.9750 | 0.9725 |
| **Transformer (DistilBERT)** | 99.27% | 0.9925 | 0.9934 | 0.9929 | 0.9923 |

### Confusion Matrices
| Model | True Positive (Spam) | False Negative (Missed Spam) | True Negative (Ham) | False Positive (False Alarm) |
|-------|----------------------|------------------------------|---------------------|------------------------------|
| **Naive Bayes** | 3288 | 184 | 3059 | 138 |
| **DNN (TensorFlow)** | 3376 | 57 | 3055 | 116 |
| **Transformer** | 3449 | 23 | 3171 | 26 |

### Model Details
1.  **Naive Bayes (`naive_bayes.ipynb`)**:
    *   Algorithm: Multinomial Naive Bayes (via Scikit-learn).
    *   Preprocessing: Tokenization, stemming (PorterStemmer), stopword removal, and CountVectorization.
    *   Performance: Good baseline with >95% accuracy.

2.  **Deep Neural Network (`dnn.ipynb`)**:
    *   Framework: TensorFlow/Keras.
    *   Architecture: Embedding layer followed by global average pooling and multiple dense layers (512, 256, 128, 64 units) with ReLU activation. Output layer uses Sigmoid activation.
    *   Performance: Improved accuracy over Naive Bayes (~97.4%).

3.  **Transformer (`transformer.ipynb`)**:
    *   Model: Fine-tuned `distilbert-base-uncased` (Hugging Face Transformers).
    *   Architecture: Pre-trained DistilBERT model with a classification head.
    *   Performance: State-of-the-art results with >99% accuracy.

## Files
*   `naive_bayes.ipynb`: Implementation of the Naive Bayes model.
*   `dnn.ipynb`: Deep Neural Network implementation using TensorFlow.
*   `transformer.ipynb`: Transformer model implementation using DistilBERT.
*   `input_data/`: Directory containing the dataset (`enron_spam_data.csv`) and resources.
