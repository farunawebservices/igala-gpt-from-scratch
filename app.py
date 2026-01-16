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
    if not os.path.exists('outputs/model_checkpoints/model_state.pth'):
        raise FileNotFoundError("Model state dict not found")
    if not os.path.exists('outputs/model_checkpoints/config.json'):
        raise FileNotFoundError("Config file not found")
    
    # Load tokenizer
    tokenizer = Tokenizer.from_file("outputs/tokenizer/igala_tokenizer.json")
    
    # Load config from JSON
    with open('outputs/model_checkpoints/config.json', 'r') as f:
        config_dict = json.load(f)
    
    config = GPTConfig(
        vocab_size=config_dict['vocab_size'],
        n_embd=config_dict['n_embd'],
        n_head=config_dict['n_head'],
        n_layer=config_dict['n_layer'],
        block_size=config_dict['block_size'],
        dropout=config_dict['dropout']
    )
    
    # Create model and load state dict
    model = IgalaGPT(config)
    state_dict = torch.load('outputs/model_checkpoints/model_state.pth', 
                           map_location='cpu',
                           weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load training logs
    with open('outputs/model_checkpoints/training_logs.json', 'r') as f:
        training_logs = json.load(f)
    
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


# Header
st.title("🇳🇬 Igala GPT: Language Model Playground")
st.markdown("**First-ever GPT model trained from scratch on Igala language**")
st.markdown("---")


if model_loaded:
    # Sidebar controls
    st.sidebar.header("⚙️ Generation Settings")
    
    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.1,
        max_value=2.0,
        value=0.8,
        step=0.1,
        help="Higher = more creative, Lower = more predictable"
    )
    
    top_k = st.sidebar.slider(
        "Top-K Sampling",
        min_value=10,
        max_value=100,
        value=40,
        step=10,
        help="Number of top tokens to sample from"
    )
    
    max_tokens = st.sidebar.slider(
        "Max New Tokens",
        min_value=10,
        max_value=100,
        value=50,
        step=10,
        help="Maximum length of generated text"
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Model Info")
    st.sidebar.metric("Parameters", f"{sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    st.sidebar.metric("Vocabulary", f"{tokenizer.get_vocab_size()}")
    st.sidebar.metric("Layers", config.n_layer)
    st.sidebar.metric("Attention Heads", config.n_head)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("✍️ Text Generation")
        
        prompt = st.text_area(
            "Enter Igala prompt:",
            value="Ọjọ la",
            height=100,
            help="Start with a few Igala words"
        )
        
        if st.button("🚀 Generate", type="primary", use_container_width=True):
            if prompt.strip():
                with st.spinner("Generating..."):
                    # Tokenize prompt
                    prompt_tokens = tokenizer.encode(prompt).ids
                    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long)
                    
                    # Generate
                    generated = model.generate(
                        prompt_tensor,
                        max_new_tokens=max_tokens,
                        temperature=temperature,
                        top_k=top_k
                    )
                    
                    # Decode
                    generated_text = tokenizer.decode(generated[0].tolist())
                    
                    # Display
                    st.success("✅ Generated text:")
                    st.info(generated_text)
                    
                    # Token count
                    st.caption(f"Generated {len(generated[0]) - len(prompt_tokens)} new tokens")
            else:
                st.warning("Please enter a prompt!")
        
        # Example prompts
        st.markdown("---")
        st.markdown("### 💡 Example Prompts")
        examples = [
            "Ọjọ la",
            "Amọnọ je",
            "Ugane ki",
            "Eju omi"
        ]
        
        for ex in examples:
            if st.button(f"Try: '{ex}'", key=ex):
                st.session_state.prompt = ex
                st.rerun()
    
    with col2:
        st.subheader("📈 Training Metrics")
        
        # Loss curve
        epochs = [log['epoch'] for log in training_logs]
        losses = [log['loss'] for log in training_logs]
        
        import plotly.graph_objects as go
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=epochs,
            y=losses,
            mode='lines+markers',
            name='Training Loss',
            line=dict(color='#1f77b4', width=3),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            title="Training Loss Over Time",
            xaxis_title="Epoch",
            yaxis_title="Loss",
            height=300,
            template="plotly_white"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Stats table
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Initial Loss", f"{losses[0]:.4f}")
        col_b.metric("Final Loss", f"{losses[-1]:.4f}")
        col_c.metric("Improvement", f"{((losses[0]-losses[-1])/losses[0]*100):.1f}%")
        
        # Perplexity calculator
        st.markdown("---")
        st.markdown("### 🎯 Perplexity Calculator")
        
        test_text = st.text_input(
            "Enter Igala text to evaluate:",
            value="Ọjọ la fọ ugane",
            help="Calculate how well the model predicts this text"
        )
        
        if st.button("Calculate Perplexity"):
            if test_text.strip():
                # Tokenize
                tokens = tokenizer.encode(test_text).ids
                if len(tokens) > 1:
                    tensor = torch.tensor([tokens], dtype=torch.long)
                    
                    # Get loss
                    with torch.no_grad():
                        _, loss = model(tensor[:, :-1], tensor[:, 1:])
                    
                    perplexity = torch.exp(loss).item()
                    
                    st.metric("Perplexity", f"{perplexity:.2f}")
                    
                    if perplexity < 10:
                        st.success("✅ Excellent! Model understands this text well.")
                    elif perplexity < 50:
                        st.info("ℹ️ Good. Model has reasonable understanding.")
                    else:
                        st.warning("⚠️ High perplexity. Text may be out of distribution.")
                else:
                    st.warning("Text too short for evaluation")
    
    # Training insights section
    st.markdown("---")
    st.subheader("📊 Training Insights")
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Load corpus stats
    with open('data/igala_corpus.txt', 'r', encoding='utf-8') as f:
        corpus = f.read()
    
    col1.metric("Training Sentences", "268")
    col2.metric("Total Characters", f"{len(corpus):,}")
    col3.metric("Total Words", f"{len(corpus.split()):,}")
    col4.metric("Training Epochs", len(training_logs))
    
    # Vocabulary stats
    st.markdown("---")
    st.subheader("📚 Vocabulary Analysis")
    
    vocab = tokenizer.get_vocab()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Special Tokens:**")
        special = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        for token in special:
            if token in vocab:
                st.code(f"{token}: ID {vocab[token]}")
    
    with col2:
        st.markdown("**Sample Tokens:**")
        sample_tokens = list(vocab.items())[:10]
        for token, idx in sample_tokens:
            st.caption(f"'{token}' → {idx}")


else:
    st.info("👆 Please ensure model files are in the correct location")
    st.markdown("""
    **Required files:**
    - `outputs/model_checkpoints/model_state.pth`
    - `outputs/model_checkpoints/config.json`
    - `outputs/model_checkpoints/training_logs.json`
    - `outputs/tokenizer/igala_tokenizer.json`
    - `data/igala_corpus.txt`
    """)


# Footer
st.markdown("---")
st.markdown("**Built by Godwin Faruna Abuh** | [Portfolio](https://your-portfolio.com) | [GitHub](https://github.com/yourusername)")
