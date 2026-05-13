# Fashion AI Assistant GUI Files

This folder contains all files needed to build a full Streamlit GUI for the Fashion AI Assistant project.

## Included Files

models/product_data.csv
- Product metadata and image paths.

models/efficientnet_features.npy
- EfficientNetB0 visual embeddings.

models/cnn_features.npy
- Custom CNN visual embeddings.

models/sbert_text_embeddings.npy
- Sentence-BERT text embeddings.

models/tfidf_features.npz
- TF-IDF sparse text features.

models/tfidf_vectorizer.pkl
- Trained TF-IDF vectorizer.

models/custom_cnn_model.keras
- Trained custom CNN model.

## Notes

The image_path column currently points to Kaggle dataset paths.
If the GUI is run outside Kaggle, copy the images folder locally and update image paths.