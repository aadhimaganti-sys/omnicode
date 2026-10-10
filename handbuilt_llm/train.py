import os
import sys
import time
import urllib.request
import torch
from model import SimpleGPT
from dataset import CharTokenizer, WordTokenizer, get_batch

# ----------------- Hyperparameters -----------------
# Configured for 10,000 steps overnight training on Raspberry Pi
max_iters = int(os.environ.get("MAX_ITERS", 10000))
eval_interval = int(os.environ.get("EVAL_INTERVAL", 500))
save_interval = int(os.environ.get("SAVE_INTERVAL", 500))
batch_size = int(os.environ.get("BATCH_SIZE", 16))
block_size = int(os.environ.get("BLOCK_SIZE", 32))
learning_rate = 1e-3
eval_iters = 10

# Optimized CPU threading: leave 1 core free so Pi stays responsive and cool overnight
if torch.cuda.is_available():
    device = "cuda"
    threads_used = 1
else:
    device = "cpu"
    # Use N-1 cores (e.g. 3 out of 4 cores) to avoid thermal throttling
    threads_used = max(1, (os.cpu_count() or 4) - 1)
    torch.set_num_threads(threads_used)

# ----------------- Data Loading -----------------
# Ensure working directory is the script directory so relative paths work
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

datasets_dir = "datasets"
all_text = []

if os.path.exists(datasets_dir) and os.path.isdir(datasets_dir):
    txt_files = sorted([f for f in os.listdir(datasets_dir) if f.endswith(".txt")])
    if txt_files:
        print("Loading training data from datasets/:")
        for fname in txt_files:
            fpath = os.path.join(datasets_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                all_text.append(content)
                print(f"  - Loaded {fname} ({len(content):,} characters)")
        text = "\n\n".join(all_text)
    else:
        text = ""
else:
    text = ""

# Fallback to input.txt if datasets/ is empty
if not text.strip():
    data_path = "input.txt"
    if not os.path.exists(data_path):
        print("Downloading TinyShakespeare dataset...")
        url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        urllib.request.urlretrieve(url, data_path)
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

# Choose your tokenizer:
tokenizer = WordTokenizer(text, max_vocab=100000)
print(f"Vocabulary size: {tokenizer.vocab_size:,} tokens")

# Train and validation splits
encoded_data = tokenizer.encode(text)

if len(encoded_data) <= block_size:
    # Repeat text if short so we have enough context
    encoded_data = encoded_data * (block_size // max(1, len(encoded_data)) + 2)

data = torch.tensor(encoded_data, dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]
if len(val_data) <= block_size:
    val_data = train_data

# ----------------- Model Setup -----------------
model = SimpleGPT(
    vocab_size=tokenizer.vocab_size,
    n_embd=64,
    n_head=4,
    n_layer=4,
    block_size=block_size,
)
model = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Resume from previous checkpoint if available
start_iter = 0
best_val_loss = float("inf")
checkpoint_path = "checkpoint_latest.pt"

if os.path.exists(checkpoint_path) and "--fresh" not in sys.argv:
    try:
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_iter = ckpt.get("iter", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from checkpoint at step {start_iter}/{max_iters} (best val loss: {best_val_loss:.4f})")
    except Exception as e:
        print(f"Notice: Could not resume checkpoint ({e}). Starting fresh.")

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, dataset in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(dataset, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

print(f"\n=======================================================")
print(f"⚡ Starting Overnight Training ({max_iters:,} steps)")
print(f"Device: {device} | CPU Cores Active: {threads_used}/{os.cpu_count() or 4}")
print(f"Checkpoints will be saved automatically to .pt files")
print(f"=======================================================\n")

start_time = time.time()
step_start = time.time()

for iter_num in range(start_iter, max_iters):
    # Periodic evaluation
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss()
        elapsed = time.time() - start_time
        steps_done = iter_num - start_iter + 1
        speed = steps_done / max(1e-5, elapsed)
        eta_seconds = (max_iters - iter_num) / max(1e-5, speed)
        eta_hours = eta_seconds / 3600.0

        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] step {iter_num:5d}/{max_iters:,} | "
            f"train: {losses['train']:.4f} | val: {losses['val']:.4f} | "
            f"speed: {speed:.1f} steps/s | ETA: {eta_hours:.1f}h"
        )

        # Save best model
        if losses["val"] < best_val_loss:
            best_val_loss = losses["val"]
            torch.save(
                {
                    "iter": iter_num,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "stoi": tokenizer.stoi,
                },
                "model_best.pt",
            )

    # Periodic checkpoint save (every 500 steps)
    if iter_num > 0 and (iter_num % save_interval == 0 or iter_num == max_iters - 1):
        torch.save(
            {
                "iter": iter_num,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
                "stoi": tokenizer.stoi,
            },
            "checkpoint_latest.pt",
        )

    # Training step
    xb, yb = get_batch(train_data, block_size, batch_size, device)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Final save
torch.save(
    {
        "iter": max_iters,
        "model_state": model.state_dict(),
        "stoi": tokenizer.stoi,
        "best_val_loss": best_val_loss,
    },
    "model_final.pt",
)
print(f"\n✓ Overnight training complete! Model saved to model_final.pt and model_best.pt\n")

print("\n" + "=" * 50)
print("TEST 1: TALKING NORMALLY (ENGLISH CONVERSATION)")
print("=" * 50)
chat_prompt = "User: Hello! Who are you?\nAssistant: "
context1 = torch.tensor([tokenizer.encode(chat_prompt)], dtype=torch.long, device=device)
gen_chat = model.generate(context1, max_new_tokens=50)[0].tolist()
print(tokenizer.decode(gen_chat))

print("\n" + "=" * 50)
print("TEST 2: CODING IN PYTHON")
print("=" * 50)
code_prompt = "def binary_search("
context2 = torch.tensor([tokenizer.encode(code_prompt)], dtype=torch.long, device=device)
gen_code = model.generate(context2, max_new_tokens=60)[0].tolist()
print(tokenizer.decode(gen_code))
