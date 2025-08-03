import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.ticker import FuncFormatter


def set_professional_style():
    """Set professional plotting style with high contrast."""
    sns.set_palette("viridis")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 16,
        'axes.labelweight': 'bold',
        'axes.titleweight': 'bold',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.2,
        'axes.grid': True,
        'grid.alpha': 0.3
    })


def create_error_analysis_plot(results_df, figsize=(16, 12)):
    """Create comprehensive error analysis plot."""
    set_professional_style()
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Model Error Analysis', fontsize=18, y=0.96, fontweight='bold')
    
    # Calculate errors if not present
    if 'Absolute Error' not in results_df.columns:
        results_df['Absolute Error'] = np.abs(results_df['Actual'] - results_df['Predicted'])
    if 'Percentage Error' not in results_df.columns:
        results_df['Percentage Error'] = 100 * results_df['Absolute Error'] / results_df['Actual']
    
    # Absolute errors with enhanced visualization
    ax1 = axes[0, 0]
    sns.histplot(results_df['Absolute Error'], kde=True, ax=ax1, 
                color='steelblue', alpha=0.7, bins=40, edgecolor='black', linewidth=0.5)
    mean_err = results_df['Absolute Error'].mean()
    median_err = results_df['Absolute Error'].median()
    ax1.axvline(mean_err, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_err:,.0f}')
    ax1.axvline(median_err, color='orange', linestyle='-.', linewidth=2, label=f'Median: {median_err:,.0f}')
    ax1.set_title('Absolute Error Distribution', pad=15)
    ax1.set_xlabel('Absolute Error')
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x:.0f}'))
    
    # Percentage errors with outlier handling
    ax2 = axes[0, 1]
    # Remove extreme outliers for better visualization
    pct_errors = results_df['Percentage Error']
    q99 = pct_errors.quantile(0.99)
    pct_errors_clean = pct_errors[pct_errors <= q99]
    
    sns.histplot(pct_errors_clean, kde=True, ax=ax2, 
                color='darkgreen', alpha=0.7, bins=40, edgecolor='black', linewidth=0.5)
    mean_pct = pct_errors_clean.mean()
    median_pct = pct_errors_clean.median()
    ax2.axvline(mean_pct, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_pct:.1f}%')
    ax2.axvline(median_pct, color='orange', linestyle='-.', linewidth=2, label=f'Median: {median_pct:.1f}%')
    ax2.set_title('Percentage Error Distribution (99th percentile)', pad=15)
    ax2.set_xlabel('Percentage Error (%)')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    
    # Actual vs Predicted with correlation
    ax3 = axes[1, 0]
    scatter = ax3.scatter(results_df['Actual'], results_df['Predicted'], 
                         alpha=0.6, s=25, c='steelblue', edgecolors='black', linewidth=0.3)
    min_val, max_val = results_df['Actual'].min(), results_df['Actual'].max()
    ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2.5, label='Perfect Prediction')
    
    # Add correlation coefficient
    corr = results_df['Actual'].corr(results_df['Predicted'])
    ax3.text(0.05, 0.95, f'R² = {corr**2:.3f}', transform=ax3.transAxes, 
            fontsize=12, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax3.set_xlabel('Actual Price')
    ax3.set_ylabel('Predicted Price')
    ax3.set_title('Actual vs Predicted Values', pad=15)
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    
    # Format axes with price formatter
    ax3.xaxis.set_major_formatter(FuncFormatter(price_formatter))
    ax3.yaxis.set_major_formatter(FuncFormatter(price_formatter))
    
    # Error by property type with enhanced visualization
    ax4 = axes[1, 1]
    # Use cleaned percentage errors for boxplot
    plot_data = results_df[results_df['Percentage Error'] <= q99].copy()
    
    sns.boxplot(x='type', y='Percentage Error', data=plot_data, ax=ax4, 
               palette='viridis', width=0.6, linewidth=1.5)
    sns.stripplot(x='type', y='Percentage Error', data=plot_data, ax=ax4,
                 color='black', alpha=0.4, size=3, jitter=True)
    
    # Add median values as text
    medians = plot_data.groupby('type')['Percentage Error'].median()
    for i, (prop_type, median_val) in enumerate(medians.items()):
        ax4.text(i, median_val + 1, f'{median_val:.1f}%', 
                ha='center', va='bottom', fontweight='bold', color='red')
    
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45, ha='right')
    ax4.set_title('Error by Property Type', pad=15)
    ax4.set_ylabel('Percentage Error (%)')
    
    plt.tight_layout()
    return fig


def price_formatter(x, pos):
    """Format price values with K/M suffixes."""
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    return f'{x:.0f}'