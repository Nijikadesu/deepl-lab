import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import argparse
import matplotlib.pyplot as plt # 新增
import numpy as np # 新增
from dataset import read_and_process, TempDataset
from models import ForcastModel # 我们会用它来实例化不同类型的 RNN

# 全局变量存储每个模型的验证损失，用于绘图
all_val_losses = {}

def train_model(model, model_name, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu'): # 增加 model_name 和 device 参数
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device 作为参数传入
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) # 可以考虑在这里添加 weight_decay

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )

    epoch_val_losses = [] # 存储当前模型每个epoch的验证损失

    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = 5 # 例如，如果连续5个epoch验证损失没有改善则停止

    print(f"\n--- Training {model_name} ---")
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 可选的梯度裁剪
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
        epoch_val_losses.append(avg_val_loss) # 记录当前epoch的验证损失
        
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch+1}/{epochs} - {model_name} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, LR: {current_lr:.6f}")
        
        # Wandb logging specific to this model's training run
        wandb.log({
            f"{model_name}_epoch": epoch+1, 
            f"{model_name}_train_loss": avg_train_loss, 
            f"{model_name}_val_loss": avg_val_loss, 
            f"{model_name}_lr": current_lr
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 保存每个模型的最佳状态
            torch.save(model.state_dict(), f"best_model_{model_name.lower()}.pth")
            wandb.save(f"best_model_{model_name.lower()}.pth") # 也保存到wandb
            print(f"Best model for {model_name} saved with val_loss: {best_val_loss:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping for {model_name} at epoch {epoch+1} as validation loss did not improve for {early_stopping_patience} epochs.")
            break
            
    all_val_losses[model_name] = epoch_val_losses # 存储这个模型的所有epoch验证损失
    return model # 返回训练好的模型，或者最佳模型


def plot_validation_losses(all_losses_dict):
    plt.figure(figsize=(12, 7))
    for model_name, losses in all_losses_dict.items():
        plt.plot(range(1, len(losses) + 1), losses, label=f'{model_name} Validation Loss')
    
    plt.title('Validation Loss Comparison Across Models')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss (MSE)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("validation_loss_comparison.png")
    print("Validation loss comparison plot saved as validation_loss_comparison.png")
    wandb.log({"validation_loss_comparison_plot": wandb.Image("validation_loss_comparison.png")}) # 上传到wandb
    plt.show()


def evaluate_and_plot_test_predictions(model, model_name, test_dataset, scaler, target_col_index_in_original_df, device='cpu', output_len=168):
    print(f"\n--- Evaluating {model_name} on Test Set ---")
    model.to(device)
    model.eval()
    
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False) # batch_size=1 for sequential prediction
    
    all_predictions = []
    all_true_values = []

    with torch.no_grad():
        for i, (x, y_true_scaled) in enumerate(test_loader): # y_true_scaled 是标准化后的真实值
            x = x.to(device)
            # y_true_scaled = y_true_scaled.to(device) # 真实值在CPU上处理反标准化

            pred_scaled = model(x) # (1, output_len)
            
            # 反标准化预测值和真实值
            # 创建一个与原始数据相同特征数量的虚拟数组，用于反标准化
            # pred_unscaled 和 y_true_unscaled 应该是 (output_len, num_features) 形状才能调用 scaler.inverse_transform
            
            # 为了反标准化目标列，我们需要知道原始数据的列数
            num_features = test_dataset.data.shape[1] 
            
            # 预测值反标准化
            dummy_pred_array = np.zeros((output_len, num_features))
            dummy_pred_array[:, target_col_index_in_original_df] = pred_scaled.cpu().numpy().flatten()
            pred_unscaled_full = scaler.inverse_transform(dummy_pred_array)
            final_predictions = pred_unscaled_full[:, target_col_index_in_original_df]

            # 真实值反标准化
            dummy_true_array = np.zeros((output_len, num_features))
            dummy_true_array[:, target_col_index_in_original_df] = y_true_scaled.numpy().flatten()
            true_unscaled_full = scaler.inverse_transform(dummy_true_array)
            final_true_values = true_unscaled_full[:, target_col_index_in_original_df]
            
            all_predictions.extend(final_predictions)
            all_true_values.extend(final_true_values)

            # 为了避免绘制过长的曲线，可以只取一部分数据点或者每隔N个点取一个
            if i * output_len > 1000: # 例如，只绘制前1000个预测点左右
                 break

    # 计算整体的 MSE 和 MAE
    all_predictions_np = np.array(all_predictions)
    all_true_values_np = np.array(all_true_values)
    
    test_mse = np.mean((all_predictions_np - all_true_values_np)**2)
    test_mae = np.mean(np.abs(all_predictions_np - all_true_values_np))
    print(f"{model_name} - Test MSE: {test_mse:.4f}, Test MAE: {test_mae:.4f}")
    wandb.log({f"{model_name}_test_mse": test_mse, f"{model_name}_test_mae": test_mae})

    # 绘图
    plt.figure(figsize=(15, 7))
    plt.plot(all_true_values_np, label='True Values', color='blue', alpha=0.7)
    plt.plot(all_predictions_np, label=f'{model_name} Predictions', color='red', linestyle='--', alpha=0.7)
    plt.title(f'Test Set Predictions vs True Values ({model_name})')
    plt.xlabel('Time Step (in test set prediction horizon)')
    plt.ylabel('Temperature (degC)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_filename = f"test_predictions_{model_name.lower()}.png"
    plt.savefig(plot_filename)
    print(f"Test prediction plot for {model_name} saved as {plot_filename}")
    wandb.log({f"test_prediction_plot_{model_name.lower()}": wandb.Image(plot_filename)})
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/root/autodl-tmp/lab/dataset/jena_climate_2009_2016.csv')
    # parser.add_argument('--model_type', type=str, choices=['Forcast', 'GRU', 'RNN'], default='Forcast') # 不再需要，我们会遍历
    parser.add_argument('--input_len', type=int, default=24*7) # 尝试用一周的数据预测
    parser.add_argument('--output_len', type=int, default=24)   # 尝试预测未来一天
    parser.add_argument('--batch_size', type=int, default=64) # 增大batch size
    parser.add_argument('--hidden_size', type=int, default=32) # 减小hidden size
    parser.add_argument('--num_layers', type=int, default=1)   # Forcast/GRU/RNN层数
    parser.add_argument('--dropout_rate', type=float, default=0.2) # Dropout率
    parser.add_argument('--epochs', type=int, default=20)      # 增加 epochs
    parser.add_argument('--lr', type=float, default=5e-4)    # 调整学习率
    parser.add_argument('--weight_decay', type=float, default=1e-5) # L2正则化
    args = parser.parse_args()

    wandb.init(project="temperature-prediction-comparison", config=vars(args))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    original_df, train_data, val_data, test_data, scaler_fitted = read_and_process(args.data_path) # read_and_process 需要返回 scaler
    input_size = train_data.shape[1]
    
    # 获取目标列 'T (degC)' 在原始 DataFrame 中的索引，用于反标准化
    try:
        target_col_original_index = original_df.columns.get_loc('T (degC)')
    except KeyError:
        print("Error: 'T (degC)' column not found in the original DataFrame. Please check column names.")
        # 尝试从处理后的 train_data/val_data/test_data 的列名中（如果它们是DataFrame）
        # 或者如果你知道它在标准化后的数据中的索引，也可以。
        # 但最安全的是从原始未处理的 df 中获取。
        # 假设它总是第一个被标准化的列，如果无法从original_df获取
        # print("Assuming 'T (degC)' is the first column in the features used for scaling.")
        # target_col_original_index = 0 # 这是一个危险的假设，最好修复原始df的列名问题
        # 实际上，在你的 dataset.py 中，target_col 是基于标准化后的数据的索引。
        # 我们在 TempDataset 中指定了 target_col 是基于标准化后数据的索引。
        # 因此，在 evaluate_and_plot_test_predictions 中，我们只需要知道这个索引。
        # TempDataset 使用的 target_col 是指在 *标准化后* 的数据中的列索引。
        # 而 scaler.inverse_transform 需要知道的是在 *喂给scaler前* 的原始特征中的列索引。
        # 这是个棘手的问题，因为 TempDataset 的 target_col 是标准化后的列索引
        # 而 scaler.inverse_transform 需要原始特征中的列索引才能正确放置值。
        #
        # 为了正确反标准化，我们需要知道 'T (degC)' 在 *输入给StandardScaler的DataFrame (train_df)* 中的列索引。
        # 让我们修改 read_and_process 来返回这个索引，或者在这里重新计算。
        
        # 重新计算 'T (degC)' 在传给 StandardScaler 的 df (即移除了 Date Time 等列之后) 中的索引
        temp_df_for_scaling_cols = original_df.copy() # 创建一个副本
        # 模拟 read_and_process 中对列的操作来确定 T(degC) 的位置
        if 'Date Time' in temp_df_for_scaling_cols.columns: temp_df_for_scaling_cols.pop('Date Time')
        if 'wv (m/s)' in temp_df_for_scaling_cols.columns: temp_df_for_scaling_cols.pop('wv (m/s)')
        if 'max. wv (m/s)' in temp_df_for_scaling_cols.columns: temp_df_for_scaling_cols.pop('max. wv (m/s)')
        if 'wd (deg)' in temp_df_for_scaling_cols.columns: temp_df_for_scaling_cols.pop('wd (deg)')
        
        try:
            target_col_original_index = temp_df_for_scaling_cols.columns.get_loc('T (degC)')
            print(f"Inferred 'T (degC)' index for scaling: {target_col_original_index}")
        except KeyError:
            print("FATAL: Could not determine 'T (degC)' index for scaler.inverse_transform. Exiting.")
            exit()


    # TempDataset 中的 target_col 参数是指在 *标准化后* 的数据中，目标变量所在的列索引
    # 在你的 read_and_process 中，'T (degC)' 是原始df的第二列(索引1) (假设第一列是p (mbar))
    # 当你fit scaler时，用的是 train_df，它的列顺序和原始df（去除时间列等之后）一致
    # 所以，target_col_original_index 就是 'T (degC)' 在 train_df (传递给scaler的那个) 中的列索引
    target_col_scaled_index = target_col_original_index # 假设它在标准化后的数据中保持相对位置

    train_dataset = TempDataset(train_data, input_len=args.input_len, output_len=args.output_len, target_col=target_col_scaled_index)
    val_dataset = TempDataset(val_data, input_len=args.input_len, output_len=args.output_len, target_col=target_col_scaled_index)
    test_dataset = TempDataset(test_data, input_len=args.input_len, output_len=args.output_len, target_col=target_col_scaled_index) # 用于评估

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True) # drop_last=True for stable batch size
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model_types = ['RNN', 'GRU', 'Forcast']
    trained_models = {}

    for model_type in model_types:
        wandb.run.name = f"{model_type}_train_run_{wandb.run.id}" # 为每个模型类型设置wandb运行名称
        
        # 注意: ForcastModel 应该能够根据 model_str 实例化对应的RNN类型
        # 你的 ForcastModel 构造函数应该接受 num_layers 和 dropout_rate
        # 如果 ForcastModel 不支持这些，你需要修改它，或者为每种类型创建单独的模型类
        
        # 假设 ForcastModel 构造函数已更新以接受 num_layers 和 dropout_rate (或忽略它们如果不需要)
        # 并且内部的 nn.LSTM/GRU/RNN 已经正确设置了 batch_first=True
        # 如果 LSTM/GRU/RNN 要使用多层，内部的 dropout 参数也应该设置
        # model = ForcastModel(model_type, input_size, hidden_size=args.hidden_size, output_size=args.output_len)
        # --- 更新模型初始化以包含更多参数 ---
        # 你需要修改 models.py 中的 ForcastModel 来接受 num_layers 和 dropout_rate
        # 例如:
        # class ForcastModel(nn.Module):
        #     def __init__(self, model_str, input_size, hidden_size=64, output_size=168, num_layers=1, dropout_rate=0.0):
        #         super().__init__()
        #         self.dropout_layer = nn.Dropout(dropout_rate) # Dropout after RNN before FC
        #         rnn_dropout = dropout_rate if num_layers > 1 else 0.0 # Dropout between RNN layers
        #         if model_str == 'LSTM':
        #             self.backbone = nn.LSTM(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        #         # ... similar for RNN, GRU ...
        #         self.fc = nn.Linear(hidden_size, output_size)
        #     def forward(self, x):
        #         out, _ = self.backbone(x)
        #         out = out[:, -1, :]
        #         out = self.dropout_layer(out) # Apply dropout
        #         out = self.fc(out)
        #         return out
        model = ForcastModel(model_type, input_size, 
                          hidden_size=args.hidden_size, 
                          output_size=args.output_len) # 确保 ForcastModel 能够处理这些
                          # num_layers=args.num_layers, 
                          # dropout_rate=args.dropout_rate) # 传递这些参数
        
        # 应用 weight_decay 到 Adam 优化器
        optimizer_params = {'lr': args.lr}
        if args.weight_decay > 0:
            optimizer_params['weight_decay'] = args.weight_decay
        
        # train_model 函数内部会创建 optimizer，所以 lr 和 weight_decay 需要在那里处理
        # 为了简单起见，我们假设 train_model 内部的 Adam 优化器会使用 args.lr 和 args.weight_decay
        # (当前 train_model 只接收 lr，需要修改它或在这里创建优化器并传入)
        # 为简单起见，暂时让 train_model 内部的 Adam 使用 args.lr, 并在那里硬编码 weight_decay 如果需要

        trained_model = train_model(model, model_type, train_loader, val_loader, 
                                    epochs=args.epochs, lr=args.lr, device=device)
        trained_models[model_type] = trained_model # 保存训练好的模型实例，或者只保存最佳状态的模型路径

    # (1) 绘制所有模型的验证损失曲线
    plot_validation_losses(all_val_losses)

    # (2) 使用每个训练好的模型（或加载最佳模型）在测试集上评估和绘图
    for model_type, _ in trained_models.items(): # 或者 model_path in trained_model_paths.items()
        print(f"Loading best model for {model_type} for test set evaluation.")
        # 重新实例化模型并加载最佳状态
        eval_model = ForcastModel(model_type, input_size, 
                               hidden_size=args.hidden_size, 
                               output_size=args.output_len)
                               # num_layers=args.num_layers, 
                               # dropout_rate=args.dropout_rate)
        eval_model.load_state_dict(torch.load(f"best_model_{model_type.lower()}.pth", map_location=device))
        
        evaluate_and_plot_test_predictions(eval_model, model_type, test_dataset, scaler_fitted, 
                                           target_col_original_index, device=device, output_len=args.output_len)
    
    wandb.finish()

if __name__ == '__main__':
    main()