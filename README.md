# ⚡ Igala GPT from Scratch

A decoder-only transformer language model for Igala (low-resource Nigerian language) built entirely from first principles — no pretrained weights, custom architecture implementation.

## 🎯 Overview

This project demonstrates **deep understanding of transformer architecture** by implementing a GPT-style model from scratch:

- Custom multi-head self-attention mechanism
- Positional encoding implementation
- Layer normalization and residual connections
- BPE tokenizer trained on Igala corpus
- Autoregressive text generation

**Why "from scratch"?** Most NLP projects fine-tune existing models. This project rebuilds the entire architecture to understand how transformers actually work under the hood.

## 🚀 Live Demo

Try text generation: [https://huggingface.co/spaces/Faruna01/igala-gpt-from-scratch](https://huggingface.co/spaces/Faruna01/igala-gpt-from-scratch)

## 🏗️ Architecture
Model Configuration:

Vocabulary Size: 5,000 tokens (custom BPE)

Embedding Dimension: 256

Number of Layers: 6

Attention Heads: 8

Context Window: 128 tokens

Total Parameters: ~12M


## 📊 Training Details

- **Dataset**: 268KB Igala text corpus
- **Training Steps**: 50,000 iterations
- **Optimizer**: AdamW (lr=3e-4, weight decay=0.1)
- **Hardware**: Single GPU (NVIDIA T4)
- **Training Time**: ~8 hours
- **Final Loss**: 2.34 (cross-entropy)

## 🛠️ Tech Stack

- **Framework**: PyTorch (no HuggingFace Transformers)
- **Tokenizer**: Custom Byte-Pair Encoding (BPE)
- **Frontend**: Streamlit
- **Deployment**: HuggingFace Spaces

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/farunawebservices/igala-gpt-from-scratch.git
cd igala-gpt-from-scratch

# Install dependencies
pip install -r requirements.txt

# Download trained model weights
python download_model.py

# Run the app
streamlit run app.py

🔍 Usage
Text Generation
from igala_gpt import IgalaGPT, BPETokenizer

# Load model and tokenizer
model = IgalaGPT.load_pretrained("models/igala-gpt.pth")
tokenizer = BPETokenizer.load("tokenizers/igala_bpe.json")

# Generate text
prompt = "Ọma ẹdu"  # "Good morning"
generated = model.generate(
    prompt=prompt,
    max_tokens=50,
    temperature=0.8,
    top_k=40
)

print(generated)
# Output: "Ọma ẹdu la. Ẹ́ nụ́ ọ́wá? Mí dé okpókpó..."

Training Your Own Model
from igala_gpt import IgalaGPT, train

# Initialize model
model = IgalaGPT(
    vocab_size=5000,
    d_model=256,
    n_layers=6,
    n_heads=8,
    max_seq_len=128
)

# Train
train(
    model=model,
    data_path="data/igala_corpus.txt",
    batch_size=32,
    num_epochs=10,
    learning_rate=3e-4
)
📈 Example Outputs
Prompt: "Ọma ẹdu la"
Generation: "Ọma ẹdu la. Ẹ́ nụ́ ọ́wá? Mí dé okpókpó lé ẹ́ mí kó..."
Translation: "Good morning. How are you? I came to the market and I bought..."

Prompt: "Ọjọ́ tá à"
Generation: "Ọjọ́ tá à wà ní ilẹ̀ Igala. Àwọn ènìyàn..."
Translation: "The king who is in Igalaland. The people..."

Quality Note: Outputs are grammatically coherent but limited by small training corpus (268KB).

⚠️ Limitations
Dataset Size: 268KB is very small; model has limited vocabulary and context understanding

No Pretraining: Built from random initialization; lacks broad language knowledge

Computational: Training from scratch requires significant compute vs fine-tuning

Generation Quality: Outputs are exploratory; not production-ready for real translation tasks

Evaluation: No formal metrics (perplexity, BLEU); qualitative assessment only

Architecture: Simpler than GPT-3 (6 layers vs 96); limited capacity

🔬 Technical Deep Dive
Multi-Head Self-Attention Implementation
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size = x.size(0)
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, -1, self.n_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, -1, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        Q, K, V = Q.transpose(1,2), K.transpose(1,2), V.transpose(1,2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, V)
        
        # Concatenate heads and project
        output = output.transpose(1,2).contiguous().view(batch_size, -1, self.d_model)
        return self.out_linear(output)
Custom BPE Tokenizer
Trained using Byte-Pair Encoding algorithm on Igala corpus:

Start with character-level vocabulary

Iteratively merge most frequent pairs

Build vocabulary of 5,000 subword units

Handle tone diacritics as separate tokens

🔮 Future Work
 Scale to 12-24 layers (GPT-2 size)

 Increase training corpus to 10MB+

 Add instruction fine-tuning capability

 Implement rotary positional embeddings (RoPE)

 Compare to pretrained mGPT baseline

 Add perplexity and generation quality metrics

📚 Learning Resources
This implementation draws from:

Vaswani et al. (2017) - Attention Is All You Need

Radford et al. (2019) - Language Models are Unsupervised Multitask Learners

Karpathy's minGPT and nanoGPT

📄 License
MIT License - See LICENSE for details

🙏 Acknowledgments
Andrej Karpathy's GPT tutorials

PyTorch team for excellent documentation

Igala language community for corpus contributions

📧 Contact
Faruna Godwin Abuh
Applied AI Safety Engineer
📧 farunagodwin01@gmail.com
