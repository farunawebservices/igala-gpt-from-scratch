import pandas as pd

# Load your corpus with correct filename
df = pd.read_csv('../igala-interpretability/data/igala_english_parallel.csv')

print(f"📊 Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Extract just Igala sentences
igala_sentences = df['igala'].dropna().tolist()

# Save as plain text (one sentence per line)
with open('data/igala_corpus.txt', 'w', encoding='utf-8') as f:
    for sentence in igala_sentences:
        f.write(sentence.strip() + '\n')

print(f"✅ Saved {len(igala_sentences)} sentences to data/igala_corpus.txt")

# Show sample
print(f"\n📝 Sample sentences:")
for i in range(min(5, len(igala_sentences))):
    print(f"  {i+1}. {igala_sentences[i]}")
