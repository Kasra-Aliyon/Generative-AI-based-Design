"""
Modularized optimization script for CCS (Carbon Capture and Storage) system analysis.
This script performs optimization and sensitivity analysis using a trained neural network model.
"""

import os
import warnings
import pickle
import numpy as np
import pandas as pd
import tensorflow.keras as keras
import pyomo.environ as pyo
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import logging
from omlt import OmltBlock, OffsetScaling
from omlt.io import load_keras_sequential
from omlt.neuralnet import FullSpaceSmoothNNFormulation
from typing import Dict, List, Tuple, Any
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

warnings.filterwarnings('ignore')

def load_config(config_path: str = "opt_config.yaml") -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Dictionary containing configuration parameters
    """
    logger.info(f"Loading configuration from {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def load_and_preprocess_data(data_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict, Dict, Dict, Dict, List, List]:
    """
    Load and preprocess the data for optimization.
    
    Args:
        data_path: Path to the input CSV file
        
    Returns:
        Tuple containing preprocessed input/output data and scaling parameters
    """
    df = pd.read_csv(data_path)
    X = df[[col for col in df.columns if col.endswith('_i')]]
    y = df[[col for col in df.columns if col.endswith('_o')]]
    
    inputs = list(X.columns)
    outputs = list(y.columns)
    
    # Calculate scaling parameters
    x_offset, x_factor = X.mean().to_dict(), X.std().to_dict()
    y_offset, y_factor = y.mean().to_dict(), y.std().to_dict()
    
    # Scale the data
    X_scaled = (X - X.mean()).divide(X.std())
    y_scaled = (y - y.mean()).divide(y.std())
    
    return X_scaled, y_scaled, x_offset, x_factor, y_offset, y_factor, inputs, outputs

def create_optimization_model(
    nn_model_path: str,
    inputs: List[str],
    outputs: List[str],
    x_offset: Dict,
    x_factor: Dict,
    y_offset: Dict,
    y_factor: Dict,
    X_scaled: pd.DataFrame
) -> Tuple[pyo.ConcreteModel, Dict[str, int]]:
    """
    Create and configure the optimization model.
    
    Args:
        nn_model_path: Path to the trained neural network model
        inputs: List of input column names
        outputs: List of output column names
        x_offset, x_factor: Input scaling parameters
        y_offset, y_factor: Output scaling parameters
        X_scaled: Scaled input data
        
    Returns:
        Tuple of optimization model and index mapping
    """
    m = pyo.ConcreteModel()
    m.CCS = OmltBlock()
    
    # Load the neural network model
    nn_CCS = keras.models.load_model(nn_model_path, compile=False)
    
    # Configure scaling
    scaler = OffsetScaling(
        offset_inputs={i: x_offset[inputs[i]] for i in range(len(inputs))},
        factor_inputs={i: x_factor[inputs[i]] for i in range(len(inputs))},
        offset_outputs={i: y_offset[outputs[i]] for i in range(len(outputs))},
        factor_outputs={i: y_factor[outputs[i]] for i in range(len(outputs))}
    )
    
    # Set input bounds
    scaled_lb = X_scaled.min()[inputs].values
    scaled_ub = X_scaled.max()[inputs].values
    scaled_input_bounds = {i: (scaled_lb[i], scaled_ub[i]) for i in range(len(inputs))}
    
    # Create network definition and build formulation
    net = load_keras_sequential(nn_CCS, scaling_object=scaler, scaled_input_bounds=scaled_input_bounds)
    m.CCS.build_formulation(FullSpaceSmoothNNFormulation(net))
    
    # Create index mapping for easy reference
    idx_map = {
        'co2cons': inputs.index('FG_co2_cons(mol_ratio)_i'),
        'lma': inputs.index('HE_left_min_apprch_i'),
        'fgt': inputs.index('FG_temp(C)_i'),
        'lat': inputs.index('LA_temp(C)_i'),
        'absh': inputs.index('Abs_ht(m)_i'),
        'slad': outputs.index('SLAD(MJ/kgco2)_o'),
        'srd': outputs.index('SRD(MJ/kgco2)_o'),
        'cap': outputs.index('Cap_eff(%)_o')
    }
    
    return m, idx_map

def run_optimization_study(
    model: pyo.ConcreteModel,
    idx_map: Dict[str, int],
    config: dict,
    inputs: List[str],
    outputs: List[str],
    co2_concentration: float
) -> pd.DataFrame:
    """
    Run the optimization study across specified ranges for a specific CO2 concentration.
    
    Args:
        model: Pyomo concrete model
        idx_map: Dictionary mapping variable names to indices
        config: Configuration dictionary
        inputs: List of input variable names
        outputs: List of output variable names
        co2_concentration: Specific CO2 concentration to analyze
    
    Returns:
        DataFrame containing optimization results
    """
    logger.info(f"Starting optimization study for CO2 concentration: {co2_concentration}")
    
    cap_range = config['optimization']['cap_eff_range']
    he_range = config['optimization']['he_range']
    constraints = config['optimization']['constraints']
    
    columns = ['CO2_concentration', 'Cap_eff(%)', 'HE_left_min_apprch'] + inputs + outputs
    results_df = pd.DataFrame(columns=columns)
    
    total_iterations = cap_range['num_points'] * he_range['num_points']
    current_iteration = 0
    
    for x in np.linspace(cap_range['start'], cap_range['stop'], cap_range['num_points']):
        for y in np.linspace(he_range['start'], he_range['stop'], he_range['num_points']):
            current_iteration += 1
            logger.info(f"Processing iteration {current_iteration}/{total_iterations} "
                       f"(Cap_eff: {x:.2f}%, HE_min_approach: {y:.2f}°C)")
            
            # Remove existing constraints and objective
            for name in ['cons1', 'cons2', 'cons3', 'cons4', 'cons5', 'cons6', 'cons7', 'obj']:
                if hasattr(model, name):
                    model.del_component(name)
            
            # Add new constraints and solve
            try:
                model.add_component('cons1', pyo.Constraint(expr=model.CCS.outputs[idx_map['cap']] == float(x)))
                model.add_component('cons2', pyo.Constraint(expr=model.CCS.inputs[idx_map['lma']] == float(y)))
                model.add_component('cons3', pyo.Constraint(expr=model.CCS.inputs[idx_map['co2cons']] == co2_concentration))
                model.add_component('cons4', pyo.Constraint(expr=model.CCS.inputs[idx_map['fgt']] == constraints['flue_gas_temp']))
                model.add_component('cons5', pyo.Constraint(expr=model.CCS.inputs[idx_map['lat']] == constraints['lean_amine_temp']))
                model.add_component('cons6', pyo.Constraint(expr=model.CCS.inputs[idx_map['absh']] == constraints['absorber_height']))
                model.add_component('cons7', pyo.Constraint(expr=model.CCS.outputs[idx_map['slad']] >= constraints['min_slad']))
                model.add_component('obj', pyo.Objective(expr=model.CCS.outputs[idx_map['srd']], sense=pyo.minimize))
                
                solver = pyo.SolverFactory('ipopt')
                status = solver.solve(model, tee=False)
                
                # Collect results
                input_results = {b: pyo.value(model.CCS.inputs[a]) for a, b in enumerate(inputs)}
                output_results = {b: pyo.value(model.CCS.outputs[a]) for a, b in enumerate(outputs)}
                row = {
                    'CO2_concentration': co2_concentration,
                    'Cap_eff(%)': x,
                    'HE_left_min_apprch': y,
                    **input_results,
                    **output_results
                }
                results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
                
            except Exception as e:
                logger.error(f"Error in optimization iteration: {e}")
                continue
    
    logger.info(f"Completed optimization study for CO2 concentration: {co2_concentration}")
    return results_df

def run_table_study(
    model: pyo.ConcreteModel,
    idx_map: Dict[str, int],
    config: dict,
    inputs: List[str],
    outputs: List[str],
    co2_concentration: float
) -> pd.DataFrame:
    """
    Run optimization study for table generation with varying absorber height,
    flue gas temperature, and lean amine temperature.
    
    Args:
        model: Pyomo concrete model
        idx_map: Dictionary mapping variable names to indices
        config: Configuration dictionary
        inputs: List of input variable names
        outputs: List of output variable names
        co2_concentration: CO2 concentration to analyze
        
    Returns:
        DataFrame containing table results
    """
    logger.info(f"Starting table study for CO2 concentration: {co2_concentration}")
    
    # Define parameter ranges for table generation
    abs_heights = config['optimization']['absorber_heights']  # Absorber heights
    fg_temps = config['optimization']['flue_gas_temps']            # Flue gas temperatures
    la_temps = config['optimization']['lean_amine_temps']          # Lean amine temperatures
    
    columns = ['Abs_ht(m)', 'FG_temp(C)', 'LA_temp(C)'] + inputs + outputs
    table_df = pd.DataFrame(columns=columns)
    
    total_iterations = len(abs_heights) * len(fg_temps) * len(la_temps)
    current_iteration = 0
    
    for abs_h in abs_heights:
        for fg_t in fg_temps:
            for la_t in la_temps:
                current_iteration += 1
                logger.info(f"Processing table iteration {current_iteration}/{total_iterations} "
                          f"(Abs_ht: {abs_h}m, FG_temp: {fg_t}°C, LA_temp: {la_t}°C)")
                
                # Remove existing constraints and objective
                for name in ['cons1', 'cons2', 'cons3', 'cons4', 'cons5', 'cons6', 'cons7', 'obj']:
                    if hasattr(model, name):
                        model.del_component(name)
                
                try:
                    # Add constraints for table generation
                    model.add_component('cons1', pyo.Constraint(expr=model.CCS.outputs[idx_map['cap']] == 90.0))  # Fixed capture efficiency
                    model.add_component('cons2', pyo.Constraint(expr=model.CCS.inputs[idx_map['lma']] == 10.0))  # Fixed minimum approach
                    model.add_component('cons3', pyo.Constraint(expr=model.CCS.inputs[idx_map['co2cons']] == co2_concentration))
                    model.add_component('cons4', pyo.Constraint(expr=model.CCS.inputs[idx_map['fgt']] == fg_t))
                    model.add_component('cons5', pyo.Constraint(expr=model.CCS.inputs[idx_map['lat']] == la_t))
                    model.add_component('cons6', pyo.Constraint(expr=model.CCS.inputs[idx_map['absh']] == abs_h))
                    model.add_component('cons7', pyo.Constraint(expr=model.CCS.outputs[idx_map['slad']] >= 0))
                    
                    model.add_component('obj', pyo.Objective(expr=model.CCS.outputs[idx_map['srd']], sense=pyo.minimize))
                    
                    # Solve
                    solver = pyo.SolverFactory('ipopt')
                    status = solver.solve(model, tee=False)
                    
                    # Collect results
                    input_results = {b: pyo.value(model.CCS.inputs[a]) for a, b in enumerate(inputs)}
                    output_results = {b: pyo.value(model.CCS.outputs[a]) for a, b in enumerate(outputs)}
                    row = {
                        'Abs_ht(m)': abs_h,
                        'FG_temp(C)': fg_t,
                        'LA_temp(C)': la_t,
                        **input_results,
                        **output_results
                    }
                    table_df = pd.concat([table_df, pd.DataFrame([row])], ignore_index=True)
                    
                except Exception as e:
                    logger.error(f"Error in table iteration: {e}")
                    continue
    
    logger.info(f"Completed table study for CO2 concentration: {co2_concentration}")
    return table_df

def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename DataFrame columns to more readable format.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with renamed columns
    """
    return df.rename(columns={
        'Cap_eff(%)': 'Capture Efficiency (%)',
        'HE_left_min_apprch': 'Minimum Temperature Approach (C)',
        'L/G (kg/kg)_i': 'L/G (kg/kg)',
        'Lean_loading(mol/mol)_i': 'Lean Loading (mol/mol)',
        'Rich_loading(mol/mol)_o': 'Rich Loading (mol/mol)',
        'SRD(MJ/kgco2)_o': 'SRD (MJ/kg CO2)',
        'SCD(MJ/kgco2)_o': 'SCD (MJ/kg CO2)',
        'SLAD(MJ/kgco2)_o': 'SLAD (MJ/kg CO2)'
    })

def extract_z_values(X_keys: List[float], Y_keys: List[float], Z_key: str, df: pd.DataFrame) -> np.ndarray:
    """
    Extract Z values from DataFrame for contour plotting.
    
    Args:
        X_keys: List of x-axis values
        Y_keys: List of y-axis values
        Z_key: Column name for z-axis values
        df: Input DataFrame
        
    Returns:
        2D numpy array of Z values
    """
    Z_values = np.zeros((len(Y_keys), len(X_keys)))
    for i, y in enumerate(Y_keys):
        for j, x in enumerate(X_keys):
            filtered_df = df[(df['Capture Efficiency (%)'] == x) & 
                           (df['Minimum Temperature Approach (C)'] == y)]
            if not filtered_df.empty:
                Z_values[i, j] = filtered_df[Z_key].values[0]
    return Z_values

def create_contour_plot(df: pd.DataFrame, plot_config: dict, variables: List[str], 
                       save_path: str = None) -> None:
    """
    Create and save contour plots.
    
    Args:
        df: Input DataFrame
        plot_config: Dictionary containing plot settings
        variables: List of variables to plot
        save_path: Optional path to save the plot
    """
    logger.info(f"Creating contour plots for variables: {variables}")
    
    # Set plot style
    font = {
        'family': plot_config['font_family'],
        'weight': plot_config['font_weight'],
        'size': plot_config['font_size']
    }
    plt.rc('font', **font)
    sns.set(style="whitegrid")
    
    try:
        # Create figure
        fig, axs = plt.subplots(1, 3, figsize=plot_config['figure_size'], 
                               dpi=plot_config['dpi'])
        axs = axs.flatten()
        
        # Get axis values
        X_keys = sorted(df['Capture Efficiency (%)'].unique())
        Y_keys = sorted(df['Minimum Temperature Approach (C)'].unique())
        
        # Create plots
        for i, Z_key in enumerate(variables):
            logger.info(f"Plotting {Z_key}")
            ax = axs[i]
            Z_values = extract_z_values(X_keys, Y_keys, Z_key, df)
            c = ax.contourf(X_keys, Y_keys, Z_values, 
                           cmap=plot_config['colormap'], 
                           levels=plot_config['levels'])
            plt.colorbar(c, ax=ax, format="%.2f")
            ax.set_title(Z_key, fontsize=plot_config['font_size']+2, 
                        fontweight=plot_config['font_weight'], 
                        fontname=plot_config['font_family'])
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            
            ax.text(0.5, -0.1, 'Capture Efficiency (%)', 
                    fontsize=plot_config['font_size'], 
                    fontname=plot_config['font_family'], 
                    ha='center', transform=ax.transAxes)
            ax.text(-0.11, 0.51, 'Minimum Temperature Approach (C)', 
                    fontsize=plot_config['font_size'], 
                    fontname=plot_config['font_family'], 
                    va='center', rotation='vertical', transform=ax.transAxes)
            
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontname(plot_config['font_family'])
            ax.tick_params(axis='both', which='major', 
                          labelsize=plot_config['font_size']-2)
        
        plt.tight_layout()
        if save_path:
            logger.info(f"Saving plot to {save_path}")
            plt.savefig(save_path, dpi=plot_config['dpi'], bbox_inches='tight')
            logger.info("Plot saved successfully")
        plt.close()
        
    except Exception as e:
        logger.error(f"Error creating contour plot: {e}")
        raise

def main():
    """Main execution function"""
    logger.info("Starting CCS optimization analysis")
    
    try:
        # Load configuration
        config = load_config()
        
        # Create results directory
        results_dir = Path(config['paths']['results_dir'])
        if not results_dir.exists():
            logger.info(f"Creating results directory at: {results_dir}")
            results_dir.mkdir(parents=True)
        
        # Load and preprocess data
        logger.info("Loading and preprocessing data")
        X_scaled, y_scaled, x_offset, x_factor, y_offset, y_factor, inputs, outputs = load_and_preprocess_data(config['paths']['data'])
        
        # Create optimization model
        logger.info("Creating optimization model")
        model, idx_map = create_optimization_model(
            config['paths']['model'], inputs, outputs, x_offset, x_factor, y_offset, y_factor, X_scaled
        )
        
        # Process each CO2 concentration
        total_concentrations = len(config['optimization']['co2_concentrations'])
        for idx, co2_conc in enumerate(config['optimization']['co2_concentrations'], 1):
            logger.info(f"\nProcessing CO2 concentration {idx}/{total_concentrations}: {co2_conc}")
            
            # Run optimization study for contour plots
            results_df = run_optimization_study(
                model, idx_map, config, inputs, outputs, co2_conc
            )
            
            # Run table study
            logger.info("Generating table results")
            table_df = run_table_study(
                model, idx_map, config, inputs, outputs, co2_conc
            )
            
            # Save results
            logger.info(f"Saving results for CO2 concentration: {co2_conc}")
            results_df.to_pickle(results_dir / f'plot_numbers_{co2_conc}.pkl')
            results_df.to_csv(results_dir / f'plot_numbers_CO2_{co2_conc}.csv', index=False)
            table_df.to_csv(results_dir / f'table_results_{co2_conc}.csv', index=False)
            
            # Create plots
            plot_df = rename_columns(results_df)
            
            logger.info("Creating energy plots")
            create_contour_plot(
                plot_df,
                config['plotting']['energy_plots'],
                config['plotting']['energy_plots']['variables'],
                save_path=str(results_dir / f'energy_plots_CO2_{co2_conc}.png')
            )
            
            logger.info("Creating operating condition plots")
            create_contour_plot(
                plot_df,
                config['plotting']['operating_plots'],
                config['plotting']['operating_plots']['variables'],
                save_path=str(results_dir / f'operating_plots_CO2_{co2_conc}.png')
            )
        
        logger.info("CCS optimization analysis completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()