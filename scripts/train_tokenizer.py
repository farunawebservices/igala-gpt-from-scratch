from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import json

# Load corpus
with open('data/igala_corpus.txt', 'r', encoding='utf-8') as f:
    corpus = f.readlines()

print(f"📚 Training tokenizer on {len(corpus)} sentences...")

# Create BPE tokenizer (byte-pair encoding)
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
tokenizer.decoder = decoders.ByteLevel()

# Train with vocab size 1500 (adjusted for 224 sentences)
trainer = trainers.BpeTrainer(
    vocab_size=1500,
    special_tokens=["<PAD>", "<BOS>", "<EOS>", "<UNK>"],
    min_frequency=2
)

tokenizer.train_from_iterator(corpus, trainer=trainer)

# Save
import os
os.makedirs('outputs/tokenizer', exist_ok=True)
tokenizer.save("outputs/tokenizer/igala_tokenizer.json")
print("✅ Tokenizer saved!")

# Test it
sample = "Ámọ̀nọ̀ jẹ ọmọ ọlọ́kọ̀"
encoded = tokenizer.encode(sample)
print(f"\n📝 Test: '{sample}'")
print(f"Tokens: {encoded.tokens}")
print(f"IDs: {encoded.ids}")

# Vocab stats
print(f"\n📊 Vocab size: {tokenizer.get_vocab_size()}")
