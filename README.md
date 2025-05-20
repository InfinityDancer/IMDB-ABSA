# IMDB Sentiment Analysis with Deep Learning and ABSA

This project explores sentiment analysis on the IMDB movie review dataset using a baseline deep learning model and an advanced Aspect-Based Sentiment Analysis (ABSA) approach.

## About the Dataset

The dataset used is the [IMDB Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/), containing 50,000 reviews labeled as positive or negative.

### Citation for the dataset:
> Maas, Andrew L., Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts.  
> *Learning Word Vectors for Sentiment Analysis*. Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).

### Columns:
- `review`: Text of the movie review.
- `sentiment`: Binary sentiment label (`positive` or `negative`).

## Objectives of the Project

- Build a baseline deep learning model using GloVe embeddings.
- Use pre-trained tools to extract aspects from reviews.
- Train an ABSA model to classify sentiment with respect to a specific aspect.
- Compare performance and insights from both models.

## Baseline Model

- Uses GloVe embeddings (`glove.6B.100d.txt`) as a static embedding layer.
- Model Architecture:
  - Embedding Layer (non-trainable)
  - Bidirectional LSTM
  - Dense Layers with Dropout
- Achieves binary classification of overall sentiment.

### Results:
| Metric       | Value   |
|--------------|---------|
| Test Loss    | 32.42%    |
| Test Accuracy| 86.67%    |

## Aspect-Based Sentiment Analysis (ABSA)

- Aspects are extracted using a pre-trained ABSA tool (e.g., `spaCy`, `transformers`, etc.).
- Only **one aspect per review** is considered for simplicity.
- The model predicts the sentiment towards that specific aspect rather than the full review.

### Example:
> *Review*: "The acting was great, but the story was weak."  
> *Aspect*: `story` → Sentiment: `negative`

## 🛠Dependencies

- Python 3.8+
- TensorFlow / Keras
- NumPy, pandas
- tqdm
- scikit-learn
- matplotlib / seaborn
- spaCy / NLTK (for ABSA)

Install dependencies:
```bash
pip install -r requirements.txt
