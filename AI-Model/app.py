import streamlit as st
import zipfile
import joblib
import os
import shutil

from utils import load_and_split_code, create_or_load_index, get_llm_response

# Set Streamlit page config
st.set_page_config(page_title="RAG-based Code Assistant", page_icon="🧠")
st.title("🧠 RAG-based Code Assistant")
st.write("Upload your Python codebase and ask questions powered by Groq + LangChain.")

# Cached model loader
@st.cache_resource
def load_model():
    model_zip_path = "model.zip"
    extract_dir = "model_dir"
    
    if not os.path.exists(extract_dir):
        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    
    return joblib.load(os.path.join(extract_dir, "model.pkl"))

# Load model once
try:
    model = load_model()
except Exception as e:
    st.error(f"⚠️ Failed to load model: {e}")
    st.stop()

# Upload .zip file
uploaded_file = st.file_uploader("📁 Upload a zipped folder of your Python codebase", type=["zip"])

# Prepare codebase folder
codebase_path = "codebase"
if uploaded_file:
    if os.path.exists(codebase_path):
        shutil.rmtree(codebase_path)
    os.makedirs(codebase_path, exist_ok=True)

    try:
        with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
            zip_ref.extractall(codebase_path)
        st.success("✅ Codebase uploaded and extracted!")
    except zipfile.BadZipFile:
        st.error("❌ Uploaded file is not a valid zip archive.")

# User input and LLM response
query = st.text_input("💬 Enter your question:")
if st.button("Get Answer") and query:
    if not uploaded_file:
        st.warning("⚠️ Please upload your codebase first.")
    else:
        with st.spinner("🔍 Indexing and retrieving relevant code..."):
            try:
                docs = load_and_split_code(codebase_path)
                db = create_or_load_index(docs)
                response = get_llm_response(query, db)
                st.success("✅ Answer:")
                st.write(response)
            except Exception as e:
                st.error(f"⚠️ Error: {str(e)}")
