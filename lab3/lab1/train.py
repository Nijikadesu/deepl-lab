import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import argparse
from dataset import read_and_process, TempDataset
from models import ForcastModel

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * x.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss = criterion(pred, y)
                total_val_loss += val_loss.item() * x.size(0)

        avg_val_loss = total_val_loss / len(val_loader.dataset)
        scheduler.step(avg_val_loss)

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")

        wandb.log({"epoch": epoch+1, "train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": current_lr})

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/lab/dataset/jena_climate_2009_2016.csv')
    parser.add_argument('--model_type', type=str, choices=['LSTM', 'GRU', 'RNN'], default='LSTM')
    parser.add_argument('--input_len', type=int, default=24)
    parser.add_argument('--output_len', type=int, default=168)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    wandb.init(project="temperature-prediction", config=vars(args))

    df, train_data, val_data, _ = read_and_process(args.data_path)
    input_size = train_data.shape[1]
    target_col = df.columns.get_loc('T (degC)')

    train_dataset = TempDataset(train_data, input_len=args.input_len, output_len=args.output_len, target_col=target_col)
    val_dataset = TempDataset(val_data, input_len=args.input_len, output_len=args.output_len, target_col=target_col)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    model = ForcastModel(args.model_type, input_size, hidden_size=args.hidden_size, output_size=args.output_len)
    train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    torch.save(model.state_dict(), f"model_{args.model_type.lower()}.pth")
    wandb.save(f"model_{args.model_type.lower()}.pth")

if __name__ == '__main__':
    main()