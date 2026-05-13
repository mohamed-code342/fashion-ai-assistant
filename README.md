# Fashion AI Assistant

## Overview

Fashion AI Assistant is a multimodal fashion recommendation system that combines computer vision and natural language processing techniques to recommend fashion products, generate outfits, and perform semantic fashion search.

The project integrates multiple deep learning and machine learning models including:

* Custom CNN
* EfficientNetB0
* TF-IDF
* Sentence-BERT

The system supports:

* Image-based product recommendation
* Text-based semantic fashion search
* Outfit generation from text
* Outfit generation from uploaded images
* Comparative analysis between different models

---

# Features

## 1. Image Recommendation

### Custom CNN Recommendation

Uses a custom-trained Convolutional Neural Network to extract visual fashion features and recommend visually similar products.

### EfficientNetB0 Recommendation

Uses pretrained EfficientNetB0 embeddings for advanced visual similarity matching.

---

## 2. Text Recommendation

### TF-IDF Fashion Search

Uses traditional NLP vectorization to retrieve fashion items using keyword similarity.

### Sentence-BERT Semantic Search

Uses transformer embeddings to understand semantic meaning and generate more intelligent recommendations.

---

## 3. Outfit Generation

### Text-to-Outfit Generation

Generates complete outfits based on text descriptions such as:

* "casual women summer outfit"
* "formal men black outfit"

### Image-to-Outfit Generation

Generates matching outfits from uploaded fashion images using EfficientNet visual understanding.

---

# Deep Learning Concepts Used

* Convolutional Neural Networks (CNNs)
* Transfer Learning
* Feature Embeddings
* EfficientNetB0
* Transformer Models
* Sentence Embeddings
* Cosine Similarity
* TF-IDF Vectorization
* Semantic Search
* Recommendation Systems
* Multimodal AI Systems

---

# Dataset

The project uses a fashion products dataset containing:

* Product images
* Product metadata
* Category labels
* Colors
* Usage styles
* Gender labels

Main categories include:

* Apparel
* Accessories
* Footwear
* Personal Care

---

# Project Structure

```bash
fashion-ai-assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
├── models/
│   ├── product_data.csv
│   ├── cnn_features.npy
│   ├── efficientnet_features.npy
│   ├── sbert_text_embeddings.npy
│   ├── tfidf_features.npz
│   ├── tfidf_vectorizer.pkl
│   └── custom_cnn_model.keras
│
└── notebooks/
    └── deep-learning-fashion.ipynb
```

---

# Installation

## 1. Clone Repository

```bash
git clone https://github.com/your-username/fashion-ai-assistant.git
cd fashion-ai-assistant
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

# Run Streamlit GUI

```bash
streamlit run app.py
```

---

# GUI Features

The Streamlit interface includes:

* Project Overview
* Custom CNN Recommendation
* EfficientNet Recommendation
* TF-IDF Text Search
* Sentence-BERT Semantic Search
* Uploaded Image Recommendation
* Text Outfit Generator
* Image Outfit Generator
* Project Files Viewer

---

# Experimental Analysis

## Image Models Comparison

Compared:

* Custom CNN
* EfficientNetB0

Evaluation Metrics:

* Average Similarity
* Same Article Type Count
* Same Color Count
* Recommendation Diversity

### Result

Custom CNN achieved higher average similarity while EfficientNet produced more category-consistent recommendations.

---

## Text Models Comparison

Compared:

* TF-IDF
* Sentence-BERT

Evaluation Metrics:

* Semantic Similarity
* Recommendation Diversity
* Color Diversity

### Result

Sentence-BERT outperformed TF-IDF in semantic understanding and recommendation quality.

---

## Outfit Recommendation Comparison

Compared:

* Rule-Based Outfit Generation
* Sentence-BERT Outfit Generation

Evaluation Metrics:

* Outfit Completeness
* Diversity
* Consistency

---

# Embedding Visualization

Used t-SNE dimensionality reduction to visualize EfficientNet embeddings.

The visualization demonstrated meaningful clustering between fashion categories, proving that the model successfully learned semantic visual representations.

---

# Error Analysis

The project also includes:

* Dataset imbalance analysis
* Color mismatch analysis
* Recommendation failure cases
* Model behavior interpretation

---

# Technologies Used

* Python
* TensorFlow / Keras
* Scikit-learn
* Sentence-Transformers
* Streamlit
* Pandas
* NumPy
* Matplotlib
* PIL

---

# Results Summary

| Task                 | Best Model                 |
| -------------------- | -------------------------- |
| Image Recommendation | Custom CNN                 |
| Text Recommendation  | Sentence-BERT              |
| Outfit Generation    | Rule-Based + Sentence-BERT |

---

# Future Improvements

Possible future extensions:

* Vision Transformers (ViTs)
* CLIP-based multimodal embeddings
* Generative AI outfit synthesis
* Personalized recommendation systems
* Real-time deployment
* Mobile application integration

---

# Authors

Deep Learning Fashion Recommendation System Project

Team Members:

* Member 1
* Member 2
* Member 3
* Member 4
* Member 5

---

# License

This project is developed for educational and research purposes.
