import os
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.config import PIPELINE_SCHEMA_DIR, JUPYTER_SCHEMA_DIR
from src.utils.helpers import get_versioned_filename


def generate_schema(df, dataset_name):
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
            'unique_values': int(df[col].nunique())
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


def schema_to_json(schema, filename=None, base_name=None, save_target='pipeline'):
    if filename is None:
        base_name = base_name or schema.get('dataset_name', 'unknown')
        schema_dir = PIPELINE_SCHEMA_DIR if save_target == 'pipeline' else JUPYTER_SCHEMA_DIR
        filename = get_versioned_filename(base_name, 'schema', schema_dir, 'json')

    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, 'w') as f:
        json.dump(schema, f, indent=2)

    return filename


def validate_against_schema(data, schema_path, log_violations=True):
    with open(schema_path, 'r') as f:
        schema = json.load(f)

    violations = []

    for col, col_schema in schema['columns'].items():
        if col not in data.columns:
            violations.append(f"Missing column: {col}")
            continue

        col_data = data[col]

        if str(col_data.dtype) != col_schema['data_type']:
            violations.append(f"Column {col}: expected {col_schema['data_type']}, got {col_data.dtype}")

        if 'numerical' in col_schema:
            num_info = col_schema['numerical']
            if col_data.min() < num_info['min']:
                violations.append(f"Column {col}: value {col_data.min()} below minimum {num_info['min']}")
            if col_data.max() > num_info['max']:
                violations.append(f"Column {col}: value {col_data.max()} above maximum {num_info['max']}")

        if 'categorical' in col_schema:
            valid_values = set(col_schema['categorical']['unique_values_list'])
            invalid_values = set(col_data.unique()) - valid_values
            if invalid_values:
                violations.append(f"Column {col}: invalid values {invalid_values}")

    if log_violations and violations:
        import logging
        logging.warning(f"Schema validation violations: {violations}")

    return len(violations) == 0, violations


class TFMALikeEvaluator:
    def __init__(self, model_pipeline):
        self.model_pipeline = model_pipeline
        self.evaluation_results = {}
        self.slice_results = {}

    def _evaluate_slice(self, X, y):
        predictions = self.model_pipeline.predict(X)
        mse = mean_squared_error(y, predictions)
        return {
            "mse": mse,
            "rmse": np.sqrt(mse),
            "mae": mean_absolute_error(y, predictions)
        }

    def evaluate(self, X_test, y_test, slice_features):
        # Overall evaluation
        self.evaluation_results = self._evaluate_slice(X_test, y_test)

        # Slice evaluation
        self.slice_results = {}
        for feature in slice_features:
            if feature not in X_test.columns:
                continue

            feature_results = {}
            for value in X_test[feature].unique():
                mask = X_test[feature] == value
                if mask.sum() >= 10:
                    feature_results[str(value)] = self._evaluate_slice(X_test[mask], y_test[mask])

            self.slice_results[feature] = feature_results

        return {"overall": self.evaluation_results, "slices": self.slice_results}

    def plot_slice_metrics(self, metric='rmse', figsize=(12, 8)):
        """TFX-style slice metrics plotting."""
        return plot_slice_metrics(self, metric, figsize)

    def get_slice_summary(self):
        return {
            "overall_evaluation": self.evaluation_results,
            "slice_evaluation": self.slice_results
        }


def plot_slice_metrics(evaluator, metric='rmse', figsize=(12, 8)):
    """TFX-style slice metrics visualization."""
    if not evaluator.slice_results:
        raise ValueError("Model must be evaluated before plotting slice metrics")

    plt.style.use('seaborn-v0_8-whitegrid')

    num_features = len(evaluator.slice_results)
    fig, axes = plt.subplots(num_features, 1, figsize=(figsize[0], figsize[1] * num_features // 2))

    if num_features == 1:
        axes = [axes]

    for i, (feature, results) in enumerate(evaluator.slice_results.items()):
        ax = axes[i]

        # Extract values and metrics
        slices = list(results.keys())
        values = [result[metric] for result in results.values()]

        # Create horizontal bar chart (TFX style)
        bars = ax.barh(slices, values, color='steelblue', alpha=0.7, edgecolor='black')

        # Add value labels
        for bar, val in zip(bars, values):
            width = bar.get_width()
            ax.text(width + width * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', ha='left', va='center', fontweight='bold')

        # Add overall metric line
        overall_metric = evaluator.evaluation_results[metric]
        ax.axvline(x=overall_metric, color='red', linestyle='--', linewidth=2,
                  label=f'Overall {metric.upper()}: {overall_metric:.3f}')

        ax.set_title(f'{metric.upper()} by {feature}', fontweight='bold', pad=15)
        ax.set_xlabel(metric.upper(), fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        ax.legend()

    plt.tight_layout()
    return fig


def visualize_statistics(schema, figsize=(16, 12)):
    """TFX-style data statistics visualization."""
    plt.style.use('seaborn-v0_8-whitegrid')

    # Calculate layout based on features
    num_features = len(schema['columns'])
    cols = min(3, num_features)
    rows = (num_features + cols - 1) // cols + 1  # +1 for summary row

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    elif cols == 1:
        axes = [[ax] for ax in axes]

    # Dataset summary
    summary_ax = axes[0][0] if cols > 1 else axes[0]
    summary_text = f"Dataset: {schema.get('dataset_name', 'Unknown')}\n"
    summary_text += f"Examples: {schema['shape']['rows']:,}\n"
    summary_text += f"Features: {schema['shape']['columns']}\n"
    summary_text += f"Generated: {schema.get('generated_at', '')[:10]}"

    summary_ax.text(0.1, 0.5, summary_text, fontsize=12, transform=summary_ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    summary_ax.set_title('Dataset Overview', fontweight='bold')
    summary_ax.axis('off')

    # Feature statistics (TFX style)
    feature_idx = 1
    for col_name, col_info in schema['columns'].items():
        if feature_idx >= rows * cols:
            break

        row_idx = feature_idx // cols
        col_idx = feature_idx % cols
        ax = axes[row_idx][col_idx] if cols > 1 else axes[row_idx]

        # Feature title
        ax.set_title(f'{col_name}', fontweight='bold', pad=10)

        if 'numerical' in col_info:
            # Numerical feature - show distribution stats
            stats = col_info['numerical']

            # Create histogram-like visualization
            values = [stats['min'], stats['mean'], stats['max']]
            labels = ['Min', 'Mean', 'Max']
            colors = ['#ff7f0e', '#2ca02c', '#d62728']

            bars = ax.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')

            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:,.0f}', ha='center', va='bottom', fontweight='bold')

            # Add statistics text
            stats_text = f"Count: {col_info['non_null_count']:,}\n"
            stats_text += f"Missing: {col_info['null_count']}\n"
            stats_text += f"Std: {stats['std']:,.1f}\n"
            stats_text += f"Zeros: {stats['zero_count']}"

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                   verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                   facecolor="white", alpha=0.8))

        else:
            # Categorical feature - show top categories
            value_counts = col_info['categorical']['value_counts']
            top_values = dict(list(value_counts.items())[:5])  # Top 5

            if top_values:
                categories = list(top_values.keys())
                counts = list(top_values.values())

                bars = ax.barh(categories, counts, color='skyblue', alpha=0.7, edgecolor='black')

                # Add count labels
                for bar, count in zip(bars, counts):
                    width = bar.get_width()
                    ax.text(width, bar.get_y() + bar.get_height()/2.,
                           f'{count}', ha='left', va='center', fontweight='bold')

                # Add statistics text
                stats_text = f"Count: {col_info['non_null_count']:,}\n"
                stats_text += f"Missing: {col_info['null_count']}\n"
                stats_text += f"Unique: {col_info['unique_values']}\n"
                stats_text += f"Top: {col_info['categorical']['most_frequent']}"

                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3",
                       facecolor="white", alpha=0.8))

        ax.grid(True, alpha=0.3)
        feature_idx += 1

    # Hide unused subplots
    for idx in range(feature_idx, rows * cols):
        row_idx = idx // cols
        col_idx = idx % cols
        if cols > 1:
            axes[row_idx][col_idx].axis('off')
        else:
            axes[row_idx].axis('off')

    plt.tight_layout()
    return fig