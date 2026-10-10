import re
import torch

class WordTokenizer:
    """Word and code tokenizer that supports up to 100,000 vocabulary items,
    preserving newlines so the model learns proper code indentation and structure."""
    def __init__(self, text: str, max_vocab: int = 100000):
        # Tokenize words, newlines, and individual code symbols
        tokens = re.findall(r"\n|\w+|[^\w\s]", text, re.UNICODE)
        
        # Count token frequencies
        counts = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
            
        # Sort by frequency and cap at max_vocab
        sorted_tokens = sorted(counts.keys(), key=lambda t: counts[t], reverse=True)
        top_tokens = sorted_tokens[: max_vocab - 2]
        
        # Special tokens: 0: <pad>, 1: <unk>
        self.stoi = {"<pad>": 0, "<unk>": 1}
        for i, tok in enumerate(top_tokens, start=2):
            self.stoi[tok] = i
            
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.vocab_size = max(len(self.stoi), max_vocab)

    def encode(self, s: str):
        tokens = re.findall(r"\n|\w+|[^\w\s]", s, re.UNICODE)
        return [self.stoi.get(tok, 1) for tok in tokens]  # 1 is <unk>

    def decode(self, token_ids):
        result = []
        for idx in token_ids:
            token = self.itos.get(idx, "<unk>")
            if token not in ("<pad>", "<unk>"):
                result.append(token)
                
        # Format code tokens back into text
        out = ""
        for i, tok in enumerate(result):
            if tok == "\n":
                out += "\n"
            elif i > 0 and result[i-1] == "\n":
                out += tok
            elif tok in ".,;:)]}>'\"":
                out += tok
            elif i > 0 and result[i-1] in "([{<'\"":
                out += tok
            else:
                out += (" " if out and not out.endswith(("\n", " ")) else "") + tok
        return out


class TiktokenTokenizer:
    """OpenAI cl100k_base BPE Tokenizer (100,256 vocab used by GPT-4)."""
    def __init__(self):
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("cl100k_base")
            self.vocab_size = self.enc.n_vocab
        except ImportError:
            raise ImportError("Please run `pip install tiktoken` to use the 100k BPE tokenizer.")

    def encode(self, s: str):
        return self.enc.encode(s)

    def decode(self, token_ids):
        return self.enc.decode(token_ids)


class CharTokenizer:
    """Simple character-level tokenizer."""
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return "".join([self.itos[i] for i in l])


def get_batch(data, block_size, batch_size, device="cpu"):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y
