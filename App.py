import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

from PIL import Image
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

from keras.applications import EfficientNetB0
from keras.applications.efficientnet import preprocess_input

from sentence_transformers import SentenceTransformer


# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="Fashion AI Assistant",
    layout="wide"
)

st.title("Fashion AI Assistant")
st.write("Multimodal Fashion Recommendation System using CNN, EfficientNet, TF-IDF, and Sentence-BERT")


# =========================
# PATHS
# =========================

MODELS_DIR = "models"

PRODUCT_DATA_PATH = os.path.join(MODELS_DIR, "product_data.csv")
CNN_FEATURES_PATH = os.path.join(MODELS_DIR, "cnn_features.npy")
EFF_FEATURES_PATH = os.path.join(MODELS_DIR, "efficientnet_features.npy")
SBERT_FEATURES_PATH = os.path.join(MODELS_DIR, "sbert_text_embeddings.npy")
TFIDF_FEATURES_PATH = os.path.join(MODELS_DIR, "tfidf_features.npz")
TFIDF_VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")


# =========================
# LOAD SAVED FILES ONLY
# =========================

@st.cache_data
def load_files():
    products = pd.read_csv(PRODUCT_DATA_PATH)

    cnn_features = np.load(CNN_FEATURES_PATH)
    efficientnet_features = np.load(EFF_FEATURES_PATH)
    sbert_embeddings = np.load(SBERT_FEATURES_PATH)

    tfidf_features = sparse.load_npz(TFIDF_FEATURES_PATH)
    tfidf_vectorizer = joblib.load(TFIDF_VECTORIZER_PATH)

    products["id"] = products["id"].astype(str)

    return (
        products,
        cnn_features,
        efficientnet_features,
        sbert_embeddings,
        tfidf_features,
        tfidf_vectorizer
    )


# =========================
# LAZY LOAD MODELS
# =========================

@st.cache_resource
def load_efficientnet_model():
    return EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_shape=(224, 224, 3),
        pooling="avg"
    )


@st.cache_resource
def load_sbert_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


products, cnn_features, efficientnet_features, sbert_embeddings, tfidf_features, tfidf_vectorizer = load_files()


# =========================
# DISPLAY HELPERS
# =========================

def show_product_card(row, score=None):
    img_path = row["image_path"]

    if os.path.exists(img_path):
        st.image(img_path, use_container_width=True)
    else:
        st.info("Image path not found locally")

    st.write(f"**ID:** {row['id']}")
    st.write(f"**Type:** {row['articleType']}")
    st.write(f"**Color:** {row['baseColour']}")
    st.write(f"**Usage:** {row['usage']}")
    st.write(f"**Gender:** {row['gender']}")

    if score is not None:
        st.write(f"**Score:** {round(float(score), 3)}")


# =========================
# RECOMMENDATION FUNCTIONS
# =========================

def recommend_by_features(product_id, features, top_n=5):
    product_id = str(product_id)

    if product_id not in products["id"].values:
        return None, None

    idx = products[products["id"] == product_id].index[0]

    scores = cosine_similarity(
        features[idx].reshape(1, -1),
        features
    ).flatten()

    top_indices = scores.argsort()[::-1][1:top_n + 1]

    results = products.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]

    return products.iloc[idx], results


def recommend_by_tfidf(query, top_n=5):
    query_vector = tfidf_vectorizer.transform([query])

    scores = cosine_similarity(
        query_vector,
        tfidf_features
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_n]

    results = products.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]

    return results


def recommend_by_sbert(query, top_n=5):
    sbert_model = load_sbert_model()

    query_embedding = sbert_model.encode(
        [query],
        convert_to_numpy=True
    )

    scores = cosine_similarity(
        query_embedding,
        sbert_embeddings
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_n]

    results = products.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]

    return results


def preprocess_uploaded_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_resized = image.resize((224, 224))

    image_array = np.array(image_resized)
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    return image, image_array


def recommend_uploaded_image(image_bytes, top_n=5):
    efficientnet_model = load_efficientnet_model()

    uploaded_image, processed_image = preprocess_uploaded_image(image_bytes)

    uploaded_features = efficientnet_model.predict(processed_image)

    scores = cosine_similarity(
        uploaded_features,
        efficientnet_features
    ).flatten()

    top_indices = scores.argsort()[::-1][:top_n]

    results = products.iloc[top_indices].copy()
    results["similarity_score"] = scores[top_indices]

    return uploaded_image, results


# =========================
# OUTFIT FUNCTIONS
# =========================

topwear_types = ["Tshirts", "Shirts", "Kurtas", "Tops", "Sweatshirts"]
bottomwear_types = ["Jeans", "Trousers", "Track Pants", "Shorts", "Skirts"]
footwear_types = ["Casual Shoes", "Sports Shoes", "Formal Shoes", "Heels", "Flats", "Sandals"]
accessory_types = ["Handbags", "Backpacks", "Watches", "Belts"]


def filter_products_by_query(query):
    query_lower = query.lower()
    filtered = products.copy()

    if "women" in query_lower or "woman" in query_lower or "female" in query_lower:
        filtered = filtered[
            filtered["gender"].str.lower().isin(["women", "girls"])
        ]

    elif "men" in query_lower or "man" in query_lower or "male" in query_lower:
        filtered = filtered[
            filtered["gender"].str.lower().isin(["men", "boys"])
        ]

    if "sports" in query_lower or "sport" in query_lower:
        filtered = filtered[
            filtered["usage"].str.lower() == "sports"
        ]

    elif "formal" in query_lower or "elegant" in query_lower:
        filtered = filtered[
            filtered["usage"].str.lower().isin(["formal", "ethnic"])
        ]

    elif "casual" in query_lower:
        filtered = filtered[
            filtered["usage"].str.lower() == "casual"
        ]

    return filtered


def select_sbert_item(query, df, article_types):
    sbert_model = load_sbert_model()

    items = df[df["articleType"].isin(article_types)].copy()

    if len(items) == 0:
        return None

    query_embedding = sbert_model.encode(
        [query],
        convert_to_numpy=True
    )

    item_embeddings = sbert_embeddings[items.index]

    scores = cosine_similarity(
        query_embedding,
        item_embeddings
    ).flatten()

    best_pos = scores.argmax()

    selected = items.iloc[best_pos].copy()
    selected["similarity_score"] = scores[best_pos]

    return selected


def generate_text_outfit(query):
    filtered = filter_products_by_query(query)

    outfit_items = [
        select_sbert_item(query, filtered, topwear_types),
        select_sbert_item(query, filtered, bottomwear_types),
        select_sbert_item(query, filtered, footwear_types),
        select_sbert_item(query, filtered, accessory_types)
    ]

    return [item for item in outfit_items if item is not None]


def generate_image_outfit(image_bytes):
    uploaded_image, recs = recommend_uploaded_image(image_bytes, top_n=1)

    base_product = recs.iloc[0]

    gender = base_product["gender"]
    usage = base_product["usage"]
    base_article = base_product["articleType"]

    filtered = products[
        (products["gender"] == gender) &
        (products["usage"] == usage)
    ]

    outfit = [base_product]

    if base_article not in topwear_types:
        items = filtered[filtered["articleType"].isin(topwear_types)]
        if len(items) > 0:
            outfit.append(items.sample(1).iloc[0])

    if base_article not in bottomwear_types:
        items = filtered[filtered["articleType"].isin(bottomwear_types)]
        if len(items) > 0:
            outfit.append(items.sample(1).iloc[0])

    if base_article not in footwear_types:
        items = filtered[filtered["articleType"].isin(footwear_types)]
        if len(items) > 0:
            outfit.append(items.sample(1).iloc[0])

    if base_article not in accessory_types:
        items = filtered[filtered["articleType"].isin(accessory_types)]
        if len(items) > 0:
            outfit.append(items.sample(1).iloc[0])

    return uploaded_image, outfit


# =========================
# SIDEBAR
# =========================

page = st.sidebar.radio(
    "Choose Feature",
    [
        "Overview",
        "Custom CNN Recommendation",
        "EfficientNet Recommendation",
        "TF-IDF Text Search",
        "Sentence-BERT Text Search",
        "Uploaded Image Recommendation",
        "Text Outfit Generator",
        "Image Outfit Generator",
        "Project Files"
    ]
)


# =========================
# PAGES
# =========================

if page == "Overview":
    st.header("Project Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Products", len(products))
    col2.metric("EfficientNet Feature Size", efficientnet_features.shape[1])
    col3.metric("Sentence-BERT Feature Size", sbert_embeddings.shape[1])

    st.subheader("Dataset Preview")
    st.dataframe(products.head(20), use_container_width=True)

    st.subheader("System Components")
    st.write("""
    - Custom CNN visual recommendation
    - EfficientNetB0 pretrained visual recommendation
    - TF-IDF text recommendation
    - Sentence-BERT semantic recommendation
    - Uploaded image recommendation
    - Text-to-outfit generation
    - Image-to-outfit generation
    """)


elif page == "Custom CNN Recommendation":
    st.header("Custom CNN Visual Recommendation")

    product_id = st.text_input("Enter Product ID", products["id"].iloc[0])
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("Recommend using CNN"):
        input_product, results = recommend_by_features(
            product_id,
            cnn_features,
            top_n
        )

        if results is None:
            st.error("Product ID not found")
        else:
            st.subheader("Input Product")
            show_product_card(input_product)

            st.subheader("Recommendations")
            cols = st.columns(top_n)

            for col, (_, row) in zip(cols, results.iterrows()):
                with col:
                    show_product_card(row, row["similarity_score"])


elif page == "EfficientNet Recommendation":
    st.header("EfficientNetB0 Visual Recommendation")

    product_id = st.text_input("Enter Product ID", products["id"].iloc[0])
    top_n = st.slider("Number of recommendations", 3, 10, 5)

    if st.button("Recommend using EfficientNet"):
        input_product, results = recommend_by_features(
            product_id,
            efficientnet_features,
            top_n
        )

        if results is None:
            st.error("Product ID not found")
        else:
            st.subheader("Input Product")
            show_product_card(input_product)

            st.subheader("Recommendations")
            cols = st.columns(top_n)

            for col, (_, row) in zip(cols, results.iterrows()):
                with col:
                    show_product_card(row, row["similarity_score"])


elif page == "TF-IDF Text Search":
    st.header("TF-IDF Text-Based Fashion Search")

    query = st.text_input("Enter text query", "casual black shoes for men")
    top_n = st.slider("Number of results", 3, 10, 5)

    if st.button("Search with TF-IDF"):
        results = recommend_by_tfidf(query, top_n)

        cols = st.columns(top_n)

        for col, (_, row) in zip(cols, results.iterrows()):
            with col:
                show_product_card(row, row["similarity_score"])


elif page == "Sentence-BERT Text Search":
    st.header("Sentence-BERT Semantic Fashion Search")

    query = st.text_input("Enter semantic query", "formal elegant women outfit")
    top_n = st.slider("Number of results", 3, 10, 5)

    if st.button("Search with Sentence-BERT"):
        with st.spinner("Loading Sentence-BERT and searching..."):
            results = recommend_by_sbert(query, top_n)

        cols = st.columns(top_n)

        for col, (_, row) in zip(cols, results.iterrows()):
            with col:
                show_product_card(row, row["similarity_score"])


elif page == "Uploaded Image Recommendation":
    st.header("Uploaded Image Recommendation using EfficientNet")

    uploaded_file = st.file_uploader(
        "Upload fashion image",
        type=["jpg", "jpeg", "png"]
    )

    top_n = st.slider("Number of results", 3, 10, 5)

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()

        if st.button("Recommend Similar Products"):
            with st.spinner("Loading EfficientNet and searching..."):
                uploaded_image, results = recommend_uploaded_image(
                    image_bytes,
                    top_n
                )

            st.subheader("Uploaded Image")
            st.image(uploaded_image, width=250)

            st.subheader("Recommended Products")
            cols = st.columns(top_n)

            for col, (_, row) in zip(cols, results.iterrows()):
                with col:
                    show_product_card(row, row["similarity_score"])


elif page == "Text Outfit Generator":
    st.header("Text-to-Outfit Generator using Sentence-BERT")

    query = st.text_input("Enter outfit description", "men formal black outfit")

    if st.button("Generate Outfit"):
        with st.spinner("Generating outfit..."):
            outfit = generate_text_outfit(query)

        if len(outfit) == 0:
            st.warning("No outfit found")
        else:
            cols = st.columns(len(outfit))

            for col, item in zip(cols, outfit):
                with col:
                    score = item.get("similarity_score", None)
                    show_product_card(item, score)


elif page == "Image Outfit Generator":
    st.header("Image-to-Outfit Generator using EfficientNet")

    uploaded_file = st.file_uploader(
        "Upload base fashion image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image_bytes = uploaded_file.read()

        if st.button("Generate Outfit From Image"):
            with st.spinner("Generating outfit from image..."):
                uploaded_image, outfit = generate_image_outfit(image_bytes)

            st.subheader("Uploaded Image")
            st.image(uploaded_image, width=250)

            st.subheader("Generated Outfit")
            cols = st.columns(len(outfit))

            for col, item in zip(cols, outfit):
                with col:
                    show_product_card(item)


elif page == "Project Files":
    st.header("Loaded Project Files")

    files = [
        "product_data.csv",
        "cnn_features.npy",
        "custom_cnn_model.keras",
        "efficientnet_features.npy",
        "sbert_text_embeddings.npy",
        "tfidf_features.npz",
        "tfidf_vectorizer.pkl"
    ]

    for file in files:
        path = os.path.join(MODELS_DIR, file)

        if os.path.exists(path):
            st.success(f"{file} found")
        else:
            st.error(f"{file} missing")
