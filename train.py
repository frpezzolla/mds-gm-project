# train.py
import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, default_data_collator
from datasets import load_dataset
from llama.model import Transformer, ModelArgs
from llama.tokenizer import Tokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    print("Loading tokenizer...")
    tokenizer = Tokenizer("model_store/tokenizer.model")
    vocab_size = tokenizer.n_words
    print(f"Tokenizer loaded. Vocab size: {vocab_size}")

    print("Loading 1% of allenai/c4 (en)...")
    dataset = load_dataset(
        "allenai/c4", "en", split="train[:1%]"
    )
    print("Dataset loaded. Number of samples:", len(dataset))


    # For demonstration, use a small batch and sequence length
    max_seq_len = 128
    batch_size = 4

    def encode(example):
        # Simple truncation, only use 'text'
        ids = tokenizer.encode(example["text"], bos=True, eos=True)
        # Truncate/pad to max_seq_len
        ids = ids[:max_seq_len]
        ids += [tokenizer.pad_id] * (max_seq_len - len(ids))
        return {"input_ids": ids}

    print("Tokenizing dataset...")
    dataset = dataset.map(encode, remove_columns=dataset.column_names)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda batch: torch.tensor([x["input_ids"] for x in batch])
    )
    print("Dataloader ready.")

    print("Building model...")
    args = ModelArgs(
        dim=512,          # Keep small for demo!
        n_layers=4,
        n_heads=8,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        max_batch_size=batch_size,
    )
    model = Transformer(args).to(DEVICE)
    print(f"Model on {DEVICE}: {sum(p.numel() for p in model.parameters()):,} parameters.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id)

    print("Starting training loop...")
    n_epochs = 2
    for epoch in range(n_epochs):
        print(f"Epoch {epoch+1}/{n_epochs}")
        model.train()
        total_loss = 0
        for step, batch in enumerate(dataloader):
            batch = batch.to(DEVICE)
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            logits = model(inputs, start_pos=0)
            logits = logits.reshape(-1, vocab_size)
            targets = targets.reshape(-1)

            loss = loss_fn(logits, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 50 == 0:
                print(f"  Step {step}: Loss {loss.item():.4f}")

        avg_loss = total_loss / (step+1)
        print(f"Epoch {epoch+1} finished. Avg loss: {avg_loss:.4f}")

    print("Training complete! You can now save or evaluate your model.")

if __name__ == "__main__":
    main()
