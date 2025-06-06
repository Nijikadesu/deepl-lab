import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
import numpy as np
import os

VOCAB_SIZE = 8000
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
SEQ_LENGTH = 100
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TEXT_FILE_PATH = "/root/autodl-tmp/lab/dataset/shakesspeare/train/tiny-shakespeare-train.txt"
TOKENIZER_PATH = "bpe-tokenizer.json"
MODEL_SAVE_PATH = "shakespeare_lstm.pth"

class ShakespeareDataset(Dataset):
    def __init__(self, text_file_path, tokenizer_path, seq_length):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.seq_length = seq_length
        
        with open(text_file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        self.encoded_text = self.tokenizer.encode(text).ids
        
        self.vocab_size = self.tokenizer.get_vocab_size()
        if self.vocab_size != VOCAB_SIZE:
            print(f"Warning: Configured VOCAB_SIZE ({VOCAB_SIZE}) does not match tokenizer's vocab size ({self.vocab_size}). Using tokenizer's.")

    def __len__(self):
        return len(self.encoded_text) - self.seq_length - 1

    def __getitem__(self, idx):
        input_seq = self.encoded_text[idx : idx + self.seq_length]
        target_seq = self.encoded_text[idx + 1 : idx + self.seq_length + 1]
        
        return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target_seq, dtype=torch.long)

class ShakespeareLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout_prob=0.2):
        super(ShakespeareLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                            batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        lstm_out = self.dropout(lstm_out)
        out = lstm_out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        out = out.view(x.size(0), x.size(1), self.vocab_size)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        h0 = weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(DEVICE)
        c0 = weight.new(self.num_layers, batch_size, self.hidden_dim).zero_().to(DEVICE)
        return (h0, c0)

def train_model(model, dataloader, num_epochs, learning_rate, device):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        model.train()
        hidden = model.init_hidden(BATCH_SIZE)
        epoch_loss = 0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if inputs.size(0) != BATCH_SIZE:
                continue

            inputs, targets = inputs.to(device), targets.to(device)
            
            hidden = tuple([h.detach() for h in hidden])

            optimizer.zero_grad()
            outputs, hidden = model(inputs, hidden)
            
            loss = criterion(outputs.reshape(-1, model.vocab_size), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average Loss: {avg_epoch_loss:.4f}")

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

def generate_text(model, tokenizer, start_string, num_generate, temperature=0.8, device='cpu'):
    model.eval()
    model.to(device)
    
    input_ids = tokenizer.encode(start_string).ids
    
    input_eval = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    generated_ids = list(input_ids)
    
    hidden = model.init_hidden(1)

    with torch.no_grad():
        if len(input_ids) > 1:
            _, hidden = model(input_eval[:, :-1], hidden)
            current_input = input_eval[:, -1].unsqueeze(0)
        else:
            current_input = input_eval 

        for _ in range(num_generate):
            output, hidden = model(current_input, hidden)
            
            output_dist = output.squeeze().div(temperature)
            probabilities = torch.softmax(output_dist, dim=-1)
            
            predicted_id = torch.multinomial(probabilities, 1).item()
            
            generated_ids.append(predicted_id)
            current_input = torch.tensor([[predicted_id]], dtype=torch.long).to(device)

    generated_text = tokenizer.decode(generated_ids)
    return generated_text

if __name__ == '__main__':
    if not os.path.exists(TOKENIZER_PATH):
        print(f"Tokenizer file not found at {TOKENIZER_PATH}. Please run the tokenizer script first.")
        exit()

    print("Loading dataset...")
    dataset = ShakespeareDataset(TEXT_FILE_PATH, TOKENIZER_PATH, SEQ_LENGTH)
    actual_vocab_size = dataset.tokenizer.get_vocab_size() 
    if actual_vocab_size != VOCAB_SIZE:
        print(f"Updating VOCAB_SIZE from config ({VOCAB_SIZE}) to actual tokenizer vocab_size ({actual_vocab_size})")
    
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    print(f"Dataset loaded. Number of sequences: {len(dataset)}, Vocab size: {actual_vocab_size}")

    model = ShakespeareLSTM(actual_vocab_size, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS)
    
    TRAIN_NEW_MODEL = True

    if TRAIN_NEW_MODEL or not os.path.exists(MODEL_SAVE_PATH):
        if not TRAIN_NEW_MODEL and not os.path.exists(MODEL_SAVE_PATH):
            print(f"Model file {MODEL_SAVE_PATH} not found. Training a new model.")
        train_model(model, dataloader, NUM_EPOCHS, LEARNING_RATE, DEVICE)
    else:
        print(f"Loading pre-trained model from {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
        model.to(DEVICE)
        print("Model loaded.")

    print("\n--- Generating Text ---")
    generation_tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    
    start_dialogue = "First Citizen:\nBefore we proceed any further, hear me speak." 

    print(f"\nSeed: '{start_dialogue}'")
    
    for temp in [0.6, 0.8, 1.0]:
        print(f"\n--- Temperature: {temp} ---")
        generated_output = generate_text(model, generation_tokenizer, start_dialogue, num_generate=200, temperature=temp, device=DEVICE)
        print(generated_output)
        print("------------------------\n")

    another_prompt = "All:\nSpeak, speak."
    print(f"\nSeed: '{another_prompt}'")
    generated_output_2 = generate_text(model, generation_tokenizer, another_prompt, num_generate=150, temperature=0.8, device=DEVICE)
    print(generated_output_2)