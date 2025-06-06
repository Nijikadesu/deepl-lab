import torch.nn as nn

# ------------------------------
# 3. 模型定义（以 LSTM 为例）
# 可切换为 GRUModel 或 RNNModel
# ------------------------------
class ForcastModel(nn.Module):
    def __init__(self, model_str, input_size, hidden_size=64, output_size=168, num_layers=1, dropout_rate=0.0):
        super().__init__()
        
        # Dropout between RNN layers (only if num_layers > 1)
        rnn_dropout_param = dropout_rate if num_layers > 1 else 0.0
        
        if model_str == 'LSTM':
            self.backbone = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                                    batch_first=True, dropout=rnn_dropout_param)
        elif model_str == 'RNN':
            self.backbone = nn.RNN(input_size, hidden_size, num_layers=num_layers, 
                                   batch_first=True, dropout=rnn_dropout_param)
        elif model_str == 'GRU':
            self.backbone = nn.GRU(input_size, hidden_size, num_layers=num_layers, 
                                   batch_first=True, dropout=rnn_dropout_param)
        else:
            raise ValueError(f"Unsupported model_str: {model_str}")
            
        self.dropout_fc = nn.Dropout(dropout_rate) # Dropout before the final FC layer
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.backbone(x) # (batch, seq_len, hidden)
        # We take the output of the last time step
        out = out[:, -1, :] 
        out = self.dropout_fc(out) # Apply dropout
        out = self.fc(out)
        return out