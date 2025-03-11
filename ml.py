"""
Machine Learning module for CCS model training and evaluation.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
import tensorflow.keras as keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

def load_config():
    """Load configuration from YAML file."""
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_data(filepath='data\\data-io.csv'):
    """
    Load and separate input/output data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        tuple: X dataframe, y dataframe, input column names, output column names
    """
    df = pd.read_csv(filepath)
    X = df[[col for col in df.columns if col.endswith('_i')]]
    y = df[[col for col in df.columns if col.endswith('_o')]]
    
    return X, y, list(X.columns), list(y.columns)

def scale_data(X, y):
    """
    Scale input and output data.
    
    Args:
        X (pd.DataFrame): Input features
        y (pd.DataFrame): Output targets
        
    Returns:
        tuple: Scaled X, scaled y, scaling parameters
    """
    x_offset, x_factor = X.mean().to_dict(), X.std().to_dict()
    y_offset, y_factor = y.mean().to_dict(), y.std().to_dict()
    
    X_scaled = (X - X.mean()).divide(X.std())
    y_scaled = (y - y.mean()).divide(y.std())
    
    scaling_params = {
        'x_offset': x_offset,
        'x_factor': x_factor,
        'y_offset': y_offset,
        'y_factor': y_factor,
        'scaled_lb': X_scaled.min().values,
        'scaled_ub': X_scaled.max().values
    }
    
    return X_scaled, y_scaled, scaling_params

def create_model(input_dim, output_dim):
    """
    Create and compile the neural network model.
    
    Args:
        input_dim (int): Number of input features
        output_dim (int): Number of output targets
        
    Returns:
        keras.Sequential: Compiled model
    """
    config = load_config()
    model_config = config['model']
    
    keras.utils.set_random_seed(42)
    
    model = Sequential(name=model_config['name'])
    
    # Add first layer
    first_layer = model_config['layers'][0]
    model.add(Dense(units=first_layer['units'], 
                   input_dim=input_dim, 
                   activation=first_layer['activation']))
    
    # Add remaining layers
    for layer in model_config['layers'][1:]:
        model.add(Dense(units=layer['units'], 
                       activation=layer['activation']))
    
    # Add output layer
    model.add(Dense(units=output_dim))
    
    optimizer = Adam() if model_config['optimizer'].lower() == 'adam' else Adam()
    model.compile(optimizer=optimizer, loss=model_config['loss'])
    
    return model

def train_model(model, X, y):
    """
    Train the neural network model.
    
    Args:
        model (keras.Sequential): Model to train
        X (pd.DataFrame): Training features
        y (pd.DataFrame): Training targets
        
    Returns:
        keras.callbacks.History: Training history
    """
    config = load_config()
    training_config = config['training']
    early_stopping_config = training_config['early_stopping']
    
    callbacks = [
        EarlyStopping(
            monitor=early_stopping_config['monitor'],
            patience=early_stopping_config['patience'],
            restore_best_weights=early_stopping_config['restore_best_weights']
        )
    ]
    
    history = model.fit(
        validation_split=training_config['validation_split'],
        x=X,
        y=y,
        epochs=training_config['epochs'],
        verbose=1,
        callbacks=callbacks
    )
    return history

def plot_training_history(history, save_path='training/training_history.png'):
    """
    Plot and save the training history.
    
    Args:
        history (keras.callbacks.History): Training history
        save_path (str): Path to save the plot
    """
    config = load_config()
    plot_config = config['plotting']
    
    plt.figure(figsize=tuple(plot_config['figure_size']))
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Mean Square Error (MSE)', fontsize=12)
    plt.title('Model Training History\nLearning Curves for Training and Validation', fontsize=14, pad=20)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=plot_config['dpi'], bbox_inches='tight')
    plt.close()

def evaluate_model(model, X_scaled, y_scaled, y_orig, outputs):
    """
    Evaluate model performance and save metrics with visualizations.
    
    Args:
        model (keras.Sequential): Trained model
        X_scaled (pd.DataFrame): Scaled input features
        y_scaled (pd.DataFrame): Scaled output targets
        y_orig (pd.DataFrame): Original (unscaled) output targets
        outputs (list): Output column names
        
    Returns:
        pd.DataFrame: Model performance metrics
    """
    config = load_config()
    plot_config = config['plotting']
    
    predictions = model.predict(X_scaled) * y_orig.std().values + y_orig.mean().values
    dfpreds = pd.DataFrame(predictions, columns=outputs)
    
    metrics = []
    for col in outputs:
        metrics.append({
            'Output': col,
            'MAE': mean_absolute_error(y_orig[col], dfpreds[col]),
            'RMSE': np.sqrt(mean_squared_error(y_orig[col], dfpreds[col])),
            'R2': r2_score(y_orig[col], dfpreds[col]),
            'MAPE': 100 * mean_absolute_percentage_error(y_orig[col], dfpreds[col])
        })
    
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('training/model_metrics.csv', index=False)
    
    create_evaluation_plots(y_orig, dfpreds, outputs, plot_config)
    
    return metrics_df

def create_evaluation_plots(y_orig, predictions, outputs, plot_config):
    """
    Create and save evaluation plots.
    
    Args:
        y_orig (pd.DataFrame): Original output values
        predictions (pd.DataFrame): Model predictions
        outputs (list): Output column names
        plot_config (dict): Plotting configuration
    """
    for col in outputs:
        plt.figure(figsize=tuple(plot_config['figure_size']))
        plt.scatter(y_orig[col], predictions[col], alpha=0.5, c='blue')
        
        min_val = min(y_orig[col].min(), predictions[col].min())
        max_val = max(y_orig[col].max(), predictions[col].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='red', linewidth=2, linestyle='--', label='Perfect Prediction')
        
        display_name = col.replace('_o', '').replace('_', ' ')
        
        plt.xlabel(f'Actual {display_name}', fontsize=12)
        plt.ylabel(f'Predicted {display_name}', fontsize=12)
        plt.title(f'{display_name}\nActual vs Predicted Values', fontsize=14, pad=20)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        filename = col.replace('(', '').replace(')', '').replace('/', '_').replace('%', 'pct')
        plt.savefig(f'training/{filename}_scatter.png', dpi=plot_config['dpi'], bbox_inches='tight')
        plt.close()

def create_training_dir():
    """Create training directory if it doesn't exist."""
    os.makedirs('training', exist_ok=True)

def load_existing_model(model_path='training/ml_model'):
    """
    Load existing model if it exists.
    
    Args:
        model_path (str): Path to the saved model
        
    Returns:
        keras.Sequential or None: Loaded model if exists, None otherwise
    """
    if os.path.exists(model_path):
        return load_model(model_path)
    return None

def main():
    """Main execution function."""
    # Create training directory
    create_training_dir()
    
    # Load and prepare data
    X, y, inputs, outputs = load_data()
    X_scaled, y_scaled, scaling_params = scale_data(X, y)
    
    # Check if model exists
    model = load_existing_model()
    
    # If no model exists, create and train a new one
    if model is None:
        print("No existing model found. Training new model...")
        model = create_model(len(inputs), len(outputs))
        history = train_model(model, X_scaled, y_scaled)
        plot_training_history(history)
        model.save('training/ml_model')
    else:
        print("Loading existing model from training/ml_model")
    
    # Evaluate model
    metrics_df = evaluate_model(model, X_scaled, y_scaled, y, outputs)
    print("\nModel Performance Metrics:")
    print(metrics_df)

if __name__ == "__main__":
    main()

