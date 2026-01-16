import streamlit as st
import torch
from tokenizers import Tokenizer
import json
import os

# Try importing model, show detailed error if missing
try:
    import sys
    sys.path.append('scripts')
    from model import IgalaGPT, GPTConfig
    MODEL_IMPORT_SUCCESS = True
except ImportError as e:
    MODEL_IMPORT_SUCCESS = False
    IMPORT_ERROR = str(e)


st.set_page_config(page_title="Igala GPT Playground", layout="wide", page_icon="🇳🇬")


# Load model and tokenizer
@st.cache_resource
def load_model():
    if not MODEL_IMPORT_SUCCESS:
        raise ImportError(f"Failed to import model: {IMPORT_ERROR}")
    
    # Check if files exist
    if not os.path.exists("outputs/tokenizer/igala_tokenizer.json"):
        raise FileNotFoundError("Tokenizer file not found")
    if not os.path.exists('outputs/model_checkpoints/igala_gpt_final.pt'):
        raise FileNotFoundError("Model checkpoint not found")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("outputs/tokenizer/igala_tokenizer.json")
    
    # Load model with weights_only=False (safe because it's our own trained model)
    try:
        checkpoint = torch.load(
            'outputs/model_checkpoints/igala_gpt_final.pt', 
            map_location='cpu',
            weights_only=False
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {str(e)}")
    
    config = checkpoint['config']
    config.vocab_size = tokenizer.get_vocab_size()
    
    model = IgalaGPT(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    training_logs = checkpoint['training_logs']
    
    return model, tokenizer, training_logs, config


# Try to load (show error if files missing)
try:
    model, tokenizer, training_logs, config = load_model()
    model_loaded = True
    st.success("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"⚠️ Failed to load model")
    st.error(f"Error: {str(e)}")
    
    # Debug info
    with st.expander("🔍 Debug Information"):
        st.write("Files in Space:")
        for root, dirs, files in os.walk('.'):
            for file in files:
                st.text(os.path.join(root, file))
    
    model_loaded = False
