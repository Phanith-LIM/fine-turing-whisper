from khmernltk import sentence_tokenize
from khmerspell import khnormal

# Read lines, strip whitespace, and remove blank lines
with open('data/8.txt', 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

text = ' '.join(lines)
encoded_text = khnormal(text)

sentences = sentence_tokenize(encoded_text)
for i, sentence in enumerate(sentences, start=1):
    print(f"{i:02d}: {sentence}")
