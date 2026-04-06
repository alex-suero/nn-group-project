import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, GAE 
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class ShallowEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ShallowEncoder, self).__init__()
        # Single Graph Convolutional Layer
        self.conv1 = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x


class ExpressionMLP(nn.Module):
    def __init__(self, input_dim=32, hidden_dim=64, output_dim=1):
        super(ExpressionMLP, self).__init__()
        
        # Define the fully connected layers
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2), # Helps prevent overfitting
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim) # No activation for regression
        )

    def forward(self, x):
        return self.network(x)
    

def train_gae(model, optimizer, x, edge_index, epochs=200):
    """
    Trains a Graph Autoencoder to reconstruct network edges.
    Returns the trained model and a list of epoch losses.
    """
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Encode the node features and graph structure into latent embedding 'z'
        z = model.encode(x, edge_index)
        
        # Calculate reconstruction loss (how well 'z' can recreate the iGRN edges)
        loss = model.recon_loss(z, edge_index)
        
        # Backpropagate and update weights
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f'Epoch: {epoch + 1:03d} | GAE Recon Loss: {loss.item():.4f}')
            
    return model, loss_history


def train_mlp(model, inputs, targets, epochs=300, lr=0.001):
    """
    Trains a Multi-Layer Perceptron (MLP) to predict expression values.
    Returns the trained model and a list of epoch losses.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.MSELoss() # Standard regression loss
    
    model.train()
    loss_history = []
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(inputs)
        
        # Calculate loss against the ground truth expression values
        loss = criterion(predictions, targets)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 20 == 0 or epoch == 0:
            print(f'MLP Epoch: {epoch + 1:03d} | MSE Loss: {loss.item():.4f}')
            
    return model, loss_history


def get_mlp_results(mlp_model, Y_train_genes, Y_test_genes, X_train_genes, 
                    X_test_genes, scaler, train_exps, all_experiments):
    # 1. Set the model to evaluation mode
    mlp_model.eval()

    # 2. Generate predictions for BOTH splits
    with torch.no_grad():
        train_predictions = mlp_model(X_train_genes).cpu().numpy()
        test_predictions = mlp_model(X_test_genes).cpu().numpy()

    # 3. Move targets to CPU
    train_targets = Y_train_genes.cpu().numpy()
    test_targets = Y_test_genes.cpu().numpy()

    # --- UNSCALING LOGIC ---
    # 4. Identify indices of the specific experiments used as targets
    train_exp_indices = [all_experiments.index(exp) for exp in train_exps]
    
    # 5. Extract means and stds for those experiments from the scaler
    exp_means = scaler.mean_[train_exp_indices]
    exp_stds = scaler.scale_[train_exp_indices]

    # 6. Step A: Reverse the StandardScaler (Z-score -> Log-TPM)
    train_predictions_log = (train_predictions * exp_stds) + exp_means
    train_targets_log = (train_targets * exp_stds) + exp_means
    
    test_predictions_log = (test_predictions * exp_stds) + exp_means
    test_targets_log = (test_targets * exp_stds) + exp_means

    # 7. Step B: Reverse the Log-transformation (Log-TPM -> Raw TPM)
    # np.expm1 calculates exp(x) - 1, which perfectly reverses np.log1p
    train_predictions_tpm = np.expm1(train_predictions_log)
    train_targets_tpm = np.expm1(train_targets_log)
    
    test_predictions_tpm = np.expm1(test_predictions_log)
    test_targets_tpm = np.expm1(test_targets_log)

    # --- METRIC CALCULATION (Scaled/Log space) ---
    # Note: These metrics are calculated on the Z-scores (the model's direct output)
    train_mse = mean_squared_error(train_targets, train_predictions)
    test_mse = mean_squared_error(test_targets, test_predictions)
    train_rho, _ = spearmanr(train_targets.flatten(), train_predictions.flatten())
    test_rho, _ = spearmanr(test_targets.flatten(), test_predictions.flatten())

    # --- METRIC CALCULATION (Real Units: Raw TPM) ---
    train_mae_tpm = mean_absolute_error(train_targets_tpm, train_predictions_tpm)
    test_mae_tpm = mean_absolute_error(test_targets_tpm, test_predictions_tpm)
    
    train_mse_tpm = mean_squared_error(train_targets_tpm, train_predictions_tpm)
    test_mse_tpm = mean_squared_error(test_targets_tpm, test_predictions_tpm)

    # 8. Construct the Results DataFrame
    results_df = pd.DataFrame({
        'Metric': ['MSE (Scaled)', 'MAE (TPM)', 'MSE (TPM)', 'Spearman Correlation'],
        'Train (Seen Genes)': [train_mse, train_mae_tpm, train_mse_tpm, train_rho],
        'Test (Unseen Genes)': [test_mse, test_mae_tpm, test_mse_tpm, test_rho]
    })

    return (results_df, (train_predictions_tpm, train_targets_tpm, 
                         test_predictions_tpm, test_targets_tpm))
    
    
def plot_expression_predictions(targets_tpm, predictions_tpm, sample_size=None, 
                                title=None):
    # 1. Flatten and Log-transform the entire dataset first
    actual_log_all = np.log1p(targets_tpm.flatten())
    pred_log_all = np.log1p(predictions_tpm.flatten())

    # 2. Calculate Spearman on the FULL dataset for accuracy
    overall_rho, _ = spearmanr(actual_log_all, pred_log_all)

    # 3. Take a random sample for the visual plot
    if sample_size is None:
        actual_plot = actual_log_all
        pred_plot = pred_log_all
    elif len(actual_log_all) > sample_size:
        # Set a seed for reproducibility
        np.random.seed(42) 
        indices = np.random.choice(len(actual_log_all), sample_size, 
                                   replace=False)
        actual_plot = actual_log_all[indices]
        pred_plot = pred_log_all[indices]
    else:
        actual_plot = actual_log_all
        pred_plot = pred_log_all
        
    pearson_r = pd.DataFrame({
        'Actual': actual_log_all,
        'Predicted': pred_log_all
    }).corr().iloc[0, 1]

    # 4. Create the plot
    plt.figure(figsize=(8, 8))
    
    # Plot the sample
    plt.scatter(actual_plot, pred_plot, alpha=0.5, s=15, color='teal', 
                edgecolor='white', linewidth=0.5, 
                label=f'n={len(actual_plot)}')
    
    # Identity Line (Perfect Prediction)
    max_val = max(actual_log_all.max(), pred_log_all.max())
    plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', lw=2, 
             label='Perfect Prediction')

    # Add Overall Metric to the plot
    plt.text(0.05, 0.92, f'Spearman ρ: {overall_rho:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    plt.text(0.05, 0.87, f'Pearson r: {pearson_r:.3f}', 
             transform=plt.gca().transAxes, fontsize=12, 
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.xlabel("Actual Expression [log(TPM + 1)]")
    plt.ylabel("Predicted Expression [log(TPM + 1)]")
    if title is not None:
        plt.title(title)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.15)
    
    plt.tight_layout()
    plt.show()