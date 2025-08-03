"""
Schema Generation and Validation Module.

This module provides functions for generating, validating, and visualizing data schemas.
It helps ensure data consistency and provides insights into dataset characteristics.
"""

# Standard library imports
import os
import json
from datetime import datetime

# Third-party imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec


# Adjust imports/paths according to your actual project structure
from src.config import PIPELINE_SCHEMA_DIR, JUPYTER_SCHEMA_DIR
from src.utils.helpers import get_versioned_filename
# Core schema functions
def generate_schema(df, dataset_name):
    """
    Generate comprehensive schema for dataset.

    Args:
        df (pandas.DataFrame): The dataframe to analyze
        dataset_name (str): Name of the dataset

    Returns:
        dict: Schema dictionary containing dataset metadata and column information
    """
    schema = {
        'dataset_name': dataset_name,
        'generated_at': datetime.now().isoformat(),
        'shape': {'rows': len(df), 'columns': len(df.columns)},
        'columns': {}
    }

    for col in df.columns:
        col_schema = {
            'data_type': str(df[col].dtype),
            'non_null_count': int(df[col].count()),
            'null_count': int(df[col].isnull().sum()),
            'unique_values': int(df[col].nunique()),
            'memory_usage_bytes': int(df[col].memory_usage(deep=True))
        }

        if df[col].dtype == 'object':
            col_schema['categorical'] = {
                'unique_values_list': df[col].unique().tolist(),
                'value_counts': df[col].value_counts().to_dict(),
                'most_frequent': df[col].mode().iloc[0] if not df[col].mode().empty else None
            }
        else:
            col_schema['numerical'] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'zero_count': int((df[col] == 0).sum()),
                'negative_count': int((df[col] < 0).sum())
            }

        schema['columns'][col] = col_schema

    return schema



def schema_to_json(
    schema,
    filename=None,
    base_name=None,
    directory=None,
    save_target='pipeline'
):
    """
    Save schema as JSON file with versioning, supporting different save targets.

    Args:
        schema (dict): Schema dictionary to save.
        filename (str, optional): Explicit path to save the JSON file. If provided,
            versioning is not used.
        base_name (str, optional): Base name for the versioned file (e.g., 'train').
            Required if filename is None.
        directory (str, optional): Manually specify the directory to save the file.
            If provided, it overrides the “save_target” selection.
        save_target (str, optional): Choice of saving location:
            - 'pipeline' -> saves to the pipeline schema directory
            - 'jupyter'  -> saves to the jupyter schema directory
            - anything else -> uses the default SCHEMA_DIR.

    Returns:
        str: The path where the schema was saved.
    """
    # If explicit filename is given, we skip versioning logic.
    if filename is None:
        if base_name is None:
            # Fallback if no base_name is provided
            base_name = schema.get('dataset_name', 'unknown')

        # Resolve directory priority:
        # 1) Use explicit 'directory' param if provided
        # 2) Otherwise choose based on save_target
        if directory is not None:
            schema_dir = directory
        else:
            if save_target.lower() == 'pipeline':
                schema_dir = PIPELINE_SCHEMA_DIR
            elif save_target.lower() == 'jupyter':
                schema_dir = JUPYTER_SCHEMA_DIR
            else:
                schema_dir = PIPELINE_SCHEMA_DIR

        # Generate a versioned filename (using project’s helper)
        filename = get_versioned_filename(
            base_name=base_name,
            file_type='schema',
            directory=schema_dir,
            extension='json'
        )

    # Create directory if it doesn’t exist
    dirname = os.path.dirname(filename)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    # Write schema to a JSON file
    with open(filename, 'w') as f:
        json.dump(schema, f, indent=2)

    return filename

# Schema validation functions
def validate_against_schema(data, schema_path, log_violations=True):
    """
    Validate data against schema and log violations.

    Args:
        data (pandas.DataFrame): Data to validate
        schema_path (str): Path to the schema JSON file
        log_violations (bool, optional): Whether to log violations. Defaults to True.

    Returns:
        tuple: (is_valid, violations) where is_valid is a boolean indicating if the data is valid,
               and violations is a list of violation messages
    """
    import logging

    # Load schema
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    violations = []

    for col, col_schema in schema['columns'].items():
        if col not in data.columns:
            violations.append(f"Missing column: {col}")
            continue

        col_data = data[col]

        # Check data type
        expected_type = col_schema['data_type']
        if str(col_data.dtype) != expected_type:
            violations.append(f"Column {col}: expected {expected_type}, got {col_data.dtype}")

        # Check numerical constraints
        if 'numerical' in col_schema:
            num_info = col_schema['numerical']
            if col_data.min() < num_info['min']:
                violations.append(f"Column {col}: value {col_data.min()} below minimum {num_info['min']}")
            if col_data.max() > num_info['max']:
                violations.append(f"Column {col}: value {col_data.max()} above maximum {num_info['max']}")

        # Check categorical constraints
        if 'categorical' in col_schema:
            valid_values = set(col_schema['categorical']['unique_values_list'])
            invalid_values = set(col_data.unique()) - valid_values
            if invalid_values:
                violations.append(f"Column {col}: invalid values {invalid_values}")

    if log_violations and violations:
        logging.warning(f"Schema validation violations: {violations}")

    return len(violations) == 0, violations

def schema_to_csv(schema, filename=None, base_name=None):
    """
    Convert schema to CSV format and save to file with versioning.

    Args:
        schema (dict): Schema dictionary to convert
        filename (str, optional): Path to save the CSV file. If provided, versioning is not used.
        base_name (str, optional): Base name for the versioned file (e.g., 'train', 'test').
            Required if filename is None.

    Returns:
        str: The path where the schema was saved.
    """
    if filename is None:
        if base_name is None:
            base_name = schema.get('dataset_name', 'unknown')

        # Generate a versioned filename
        filename = get_versioned_filename(
            base_name=base_name,
            file_type='schema',
            directory=SCHEMA_DIR,
            extension='csv'
        )

    # Ensure directory exists
    dirname = os.path.dirname(filename)
    if dirname:  # Only create directory if dirname is not empty
        os.makedirs(dirname, exist_ok=True)

    rows = []
    for col_name, col_info in schema['columns'].items():
        row = {
            'column': col_name,
            'data_type': col_info['data_type'],
            'non_null_count': col_info['non_null_count'],
            'null_count': col_info['null_count'],
            'unique_values': col_info['unique_values']
        }

        if 'categorical' in col_info:
            row['min_value'] = None
            row['max_value'] = None
            row['mean'] = None
            row['most_frequent'] = col_info['categorical']['most_frequent']
        else:
            row['min_value'] = col_info['numerical']['min']
            row['max_value'] = col_info['numerical']['max']
            row['mean'] = col_info['numerical']['mean']
            row['most_frequent'] = None

        rows.append(row)

    pd.DataFrame(rows).to_csv(filename, index=False)

    return filename

def generate_and_save_schemas(train_path=None, test_path=None, output_dir=None):
    """
    Generate and save schemas for training and test datasets with configurable paths.
    Uses versioned filenames based on the DATA_VERSION.

    Args:
        train_path (str, optional): Path to training data CSV.
            Defaults to using the path from config.
        test_path (str, optional): Path to test data CSV.
            Defaults to using the path from config.
        output_dir (str, optional): Directory to save schema files.
            Defaults to SCHEMA_DIR from config.

    Returns:
        tuple: (train_schema, test_schema) containing the generated schema dictionaries
    """
    from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH

    # Use default paths if not provided
    if train_path is None:
        train_path = TRAIN_DATA_PATH
    if test_path is None:
        test_path = TEST_DATA_PATH
    if output_dir is None:
        output_dir = SCHEMA_DIR

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Generate schemas
    train_schema = generate_schema(train_data, 'training_data')
    test_schema = generate_schema(test_data, 'test_data')

    # Save schemas with versioning
    train_json_path = schema_to_json(train_schema, base_name='train')
    test_json_path = schema_to_json(test_schema, base_name='test')

    train_csv_path = schema_to_csv(train_schema, base_name='train')
    test_csv_path = schema_to_csv(test_schema, base_name='test')

    print("✅ Schema files generated successfully:")
    print(f"  - {train_json_path}")
    print(f"  - {test_json_path}")
    print(f"  - {train_csv_path}")
    print(f"  - {test_csv_path}")

    return train_schema, test_schema

def load_schema(schema_path):
    """
    Load schema from JSON file.

    Args:
        schema_path (str): Path to the schema JSON file

    Returns:
        dict: Loaded schema dictionary
    """
    with open(schema_path, 'r') as f:
        return json.load(f)


# Visualization functions and helpers
def plot_numerical_group(fig, gs, current_row, group_df, title_suffix=""):
    """
    Plot a group of numerical features showing min, mean, and max values.

    Args:
        fig (matplotlib.figure.Figure): Figure object to add the plot to
        gs (matplotlib.gridspec.GridSpec): GridSpec object for subplot positioning
        current_row (int): Current row index in the GridSpec
        group_df (pandas.DataFrame): DataFrame containing numerical feature statistics
        title_suffix (str, optional): Suffix to add to the plot title

    Returns:
        int: Updated current_row index
    """
    if group_df.empty:
        return current_row  # Skip if group is empty

    # Create a new subplot for this group
    ax_num_dist = fig.add_subplot(gs[current_row, 0])
    current_row += 1

    # Plot min, mean, max as a horizontal bar chart
    group_df_melted = pd.melt(
        group_df,
        id_vars=['column'],
        value_vars=['min', 'mean', 'max'],
        var_name='stat',
        value_name='value'
    )

    # Use different colors for min, mean, max with better contrast
    palette = {'min': '#3498db', 'mean': '#2ecc71', 'max': '#e74c3c'}

    bars = sns.barplot(
        x='value',
        y='column',
        hue='stat',
        data=group_df_melted,
        palette=palette,
        ax=ax_num_dist,
        edgecolor='black',
        linewidth=1
    )

    # Format x-axis with appropriate scale
    max_value = group_df_melted['value'].max()
    if max_value > 1000000:
        ax_num_dist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000000:.1f}M'))
    elif max_value > 1000:
        ax_num_dist.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.1f}K'))

    ax_num_dist.set_title(f'Numerical Features - Min, Mean, Max {title_suffix}',
                        fontsize=18, fontweight='bold', pad=15)
    ax_num_dist.set_xlabel('Value', fontsize=14, labelpad=10)
    ax_num_dist.set_ylabel('Column', fontsize=14, labelpad=10)
    ax_num_dist.tick_params(axis='both', which='major', labelsize=12)
    ax_num_dist.grid(True, axis='x', linestyle='--', alpha=0.7)
    ax_num_dist.legend(title='Statistic', fontsize=12, title_fontsize=14)

    return current_row


# Main execution function
def main():
    """
    Main function to generate schemas when run as script.

    This function is called when the module is run directly as a script.
    It generates schemas for the default data paths from the config.
    """
    from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, SCHEMA_DIR

    # Use the paths from config
    generate_and_save_schemas(TRAIN_DATA_PATH, TEST_DATA_PATH, SCHEMA_DIR)

def visualize_statistics(schema, figsize=(24, 16)):
    """
    Visualize statistics from a schema, similar to tfdv.visualize_statistics.
    Provides a clear and focused view of dataset statistics without redundant information.

    Args:
        schema: Schema dictionary generated by generate_schema function
        figsize: Size of the figure (width, height) in inches

    Returns:
        matplotlib figure with visualizations
    """
    # Set style for better readability
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("talk")  # Larger context for better readability

    # Create figure and grid layout with one plot per row
    # Calculate total number of plots needed
    numerical_cols = [col for col, info in schema['columns'].items() if 'numerical' in info]
    categorical_cols = [col for col, info in schema['columns'].items() if 'categorical' in info]

    # Start with 1 for the summary
    total_plots = 1

    # Add plots for numerical features
    if len(numerical_cols) > 0:
        # For numerical feature groups (up to 3 groups)
        if len(numerical_cols) > 1:
            # Check if we need to create multiple groups
            num_stats = []
            for col in numerical_cols:
                info = schema['columns'][col]['numerical']
                num_stats.append({
                    'column': col,
                    'max': info['max']
                })

            # Sort by max value
            num_df = pd.DataFrame(num_stats).sort_values('max', ascending=False)

            # Calculate the ratio between the largest and second largest max value
            if len(num_df) > 1:
                max_ratio = num_df.iloc[0]['max'] / max(num_df.iloc[1]['max'], 1)

                if max_ratio > 10:
                    # At least 2 groups
                    num_groups = 2

                    # Check if we need a third group
                    if len(num_df) > 2:
                        remaining_ratio = num_df.iloc[1]['max'] / max(num_df.iloc[2]['max'], 1)
                        if remaining_ratio > 10:
                            num_groups = 3
                else:
                    # Split into two groups at the median
                    num_groups = 2
            else:
                num_groups = 1
        else:
            num_groups = 1

        total_plots += num_groups  # Add plots for numerical feature groups
        total_plots += min(6, len(numerical_cols))  # Box plots (up to 6)

    # Add plots for categorical features (up to 6)
    total_plots += min(6, len(categorical_cols))

    fig = plt.figure(figsize=(24, total_plots * 5))  # Adjust height based on number of plots
    gs = GridSpec(total_plots, 1, figure=fig, hspace=0.4)

    # Dataset summary - row 0
    current_row = 0
    ax_summary = fig.add_subplot(gs[current_row, 0])
    ax_summary.axis('off')
    dataset_name = schema.get('dataset_name', 'Dataset')
    rows = schema['shape']['rows']
    cols = schema['shape']['columns']
    generated_at = schema.get('generated_at', 'Unknown')

    summary_text = (
        f"Dataset: {dataset_name}\n"
        f"Rows: {rows:,}\n"
        f"Columns: {cols}\n"
        f"Generated: {generated_at.split('T')[0] if 'T' in generated_at else generated_at}\n\n"
    )

    # Count data types
    data_types = {}
    for col, info in schema['columns'].items():
        dtype = info['data_type']
        data_types[dtype] = data_types.get(dtype, 0) + 1

    summary_text += "Data Types:\n"
    for dtype, count in data_types.items():
        summary_text += f"  - {dtype}: {count} columns\n"

    ax_summary.text(0.5, 0.5, summary_text, va='center', ha='center', fontsize=16,
                   fontweight='medium', bbox=dict(boxstyle="round,pad=0.5",
                                               facecolor="lightyellow", alpha=0.5))
    ax_summary.set_title('Dataset Summary', fontsize=22, fontweight='bold', pad=20)
    current_row += 1

    # Numerical features

    if numerical_cols:
        # Create a DataFrame for numerical stats
        num_stats = []
        for col in numerical_cols:
            info = schema['columns'][col]['numerical']
            num_stats.append({
                'column': col,
                'min': info['min'],
                'max': info['max'],
                'mean': info['mean'],
                'std': info['std']
            })

        num_df = pd.DataFrame(num_stats)

        # Group numerical features based on their maximum values
        # This helps to separate features with very large values from those with smaller values
        # for better visualization

        # Sort by max value
        num_df = num_df.sort_values('max', ascending=False)

        # Determine grouping based on value ranges
        if len(num_df) > 1:
            # Calculate the ratio between the largest and second largest max value
            max_ratio = num_df.iloc[0]['max'] / max(num_df.iloc[1]['max'], 1)  # Avoid division by zero

            # If the largest value is significantly larger (10x) than the second largest,
            # create separate groups
            if max_ratio > 10:
                # Group 1: Features with very large values (top feature)
                group1 = num_df.iloc[[0]]
                # Group 2: Rest of the features
                group2 = num_df.iloc[1:]

                # If there's a big gap in the remaining features, create a third group
                if len(group2) > 1:
                    remaining_ratio = group2.iloc[0]['max'] / max(group2.iloc[1]['max'], 1)
                    if remaining_ratio > 10:
                        # Group 2: Second largest feature
                        group2 = group2.iloc[[0]]
                        # Group 3: Rest of the features
                        group3 = num_df.iloc[2:]
                    else:
                        group3 = pd.DataFrame()  # Empty DataFrame if no third group
                else:
                    group3 = pd.DataFrame()  # Empty DataFrame if only two features
            else:
                # If values are more evenly distributed, split into two groups at the median
                median_idx = len(num_df) // 2
                group1 = num_df.iloc[:median_idx]
                group2 = num_df.iloc[median_idx:]
                group3 = pd.DataFrame()  # Empty DataFrame if no third group
        else:
            # If there's only one feature, put it in group1
            group1 = num_df
            group2 = pd.DataFrame()  # Empty DataFrame
            group3 = pd.DataFrame()  # Empty DataFrame

        # Use the helper function to plot each group

        # Plot each group with appropriate title
        if not group3.empty:
            current_row = plot_numerical_group(fig, gs, current_row, group1, "(Largest Values)")
            current_row = plot_numerical_group(fig, gs, current_row, group2, "(Medium Values)")
            current_row = plot_numerical_group(fig, gs, current_row, group3, "(Smallest Values)")
        elif not group2.empty:
            current_row = plot_numerical_group(fig, gs, current_row, group1, "(Larger Values)")
            current_row = plot_numerical_group(fig, gs, current_row, group2, "(Smaller Values)")
        else:
            current_row = plot_numerical_group(fig, gs, current_row, group1, "")

        # Box plots for numerical features (up to 6) - one per row
        for i, col in enumerate(numerical_cols[:6]):
            # Each box plot gets its own row
            ax = fig.add_subplot(gs[current_row, 0])
            current_row += 1

            # Get stats for this column
            stats = schema['columns'][col]['numerical']

            # Create a box plot using the statistics
            box_stats = {
                'whislo': stats['min'],
                'q1': stats['min'] + (stats['max'] - stats['min']) * 0.25,  # Approximation
                'med': stats['mean'],  # Using mean as approximation for median
                'q3': stats['min'] + (stats['max'] - stats['min']) * 0.75,  # Approximation
                'whishi': stats['max'],
                'fliers': []  # No outliers in this simplified view
            }

            # Use a more visible color for the box plot
            ax.bxp([box_stats], showfliers=False, vert=False,
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=2),
                  medianprops=dict(color='darkred', linewidth=2),
                  whiskerprops=dict(linewidth=2),
                  capprops=dict(linewidth=2))

            ax.set_title(f'{col}', fontsize=16, fontweight='bold', pad=10)
            ax.set_xlabel('Value', fontsize=14, labelpad=10)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)

            # Format x-axis with appropriate scale
            max_val = stats['max']
            if max_val > 1000000:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000000:.1f}M'))
            elif max_val > 1000:
                ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/1000:.1f}K'))

            # Add stats as text with better formatting
            stats_text = (
                f"Min: {stats['min']:,.2f}\n"
                f"Max: {stats['max']:,.2f}\n"
                f"Mean: {stats['mean']:,.2f}\n"
                f"Std: {stats['std']:,.2f}\n"
                f"Zeros: {stats['zero_count']}"
            )
            ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                   va='top', fontsize=12, fontweight='medium',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           alpha=0.9, edgecolor='gray'))

    # Categorical features

    if categorical_cols:
        # For each categorical feature (up to 6), show top categories - one per row
        for i, col in enumerate(categorical_cols[:6]):
            # Each categorical plot gets its own row
            ax = fig.add_subplot(gs[current_row, 0])
            current_row += 1

            # Get value counts
            value_counts = schema['columns'][col]['categorical']['value_counts']

            # Convert to DataFrame and sort
            cat_df = pd.DataFrame({
                'category': list(value_counts.keys()),
                'count': list(value_counts.values())
            }).sort_values('count', ascending=False).head(10)  # Top 10 categories

            # Plot with better colors and labels
            bars = sns.barplot(x='count', y='category', data=cat_df, ax=ax,
                             palette='viridis', edgecolor='black', linewidth=1)

            # Add value labels to bars
            for i, p in enumerate(bars.patches):
                width = p.get_width()
                ax.text(width + 0.5, p.get_y() + p.get_height()/2,
                      f'{width:,.0f}', ha='left', va='center', fontsize=11, fontweight='bold')

            ax.set_title(f'{col} (Top Categories)', fontsize=16, fontweight='bold', pad=10)
            ax.set_xlabel('Count', fontsize=14, labelpad=10)
            ax.set_ylabel('', fontsize=14)
            ax.tick_params(axis='both', which='major', labelsize=12)
            ax.grid(True, axis='x', linestyle='--', alpha=0.7)

            # Add stats with better formatting
            unique = schema['columns'][col]['unique_values']
            most_freq = schema['columns'][col]['categorical']['most_frequent']

            # Truncate most_freq if it's too long
            if most_freq and isinstance(most_freq, str) and len(most_freq) > 20:
                most_freq = most_freq[:17] + "..."

            stats_text = f"Unique values: {unique}\nMost frequent: {most_freq}"
            ax.text(0.05, 0.05, stats_text, transform=ax.transAxes,
                   va='bottom', fontsize=12, fontweight='medium',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                           alpha=0.9, edgecolor='gray'))

    # Use subplots_adjust instead of tight_layout for better compatibility
    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.3, hspace=0.4)
    return fig

if __name__ == '__main__':
    main()
