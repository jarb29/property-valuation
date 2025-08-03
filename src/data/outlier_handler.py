import pandas as pd
import os

# Adjust these imports according to your actual project structure
from src.config import OUTPUT_DIR, PIPELINE_DATA_DIR, JUPYTER_DATA_DIR
from src.utils.helpers import get_versioned_filename

def separate_outliers_and_save(data_path=None, save_target='pipeline'):
    """
    Separates data into outliers and non-outliers using a hierarchical grouping
    approach with area binning. Adds an extra level of grouping by area bins.
    Uses versioned filenames based on the DATA_VERSION.

    Args:
        data_path (str, optional): Path to the input data file.
            Defaults to the training data path from config if not provided.
        save_target (str, optional): Where to save the output.
            - 'pipeline' -> saves to the pipeline output directory
            - 'jupyter' -> saves to the jupyter output directory
            - (anything else) -> uses the default OUTPUT_DIR

    Returns:
        tuple: (data_without_outliers, data_with_outliers) - DataFrames without
               and with outliers.
    """
    # Resolve where to save
    if save_target.lower() == 'pipeline':
        output_dir = PIPELINE_DATA_DIR
    elif save_target.lower() == 'jupyter':
        output_dir = JUPYTER_DATA_DIR
    else:
        output_dir = OUTPUT_DIR

    # If no data path is provided, use the default training data path
    if data_path is None:
        from src.config import TRAIN_DATA_PATH
        data_path = TRAIN_DATA_PATH

    # Load data
    df = pd.read_csv(data_path)

    # Remove zero values
    df_clean = df[~(df == 0).any(axis=1)].copy()

    # Flag for outliers
    df_clean['is_outlier'] = False

    # Group by sector -> type -> n_rooms -> n_bathroom
    for (sector, prop_type, rooms, bathrooms), group in df_clean.groupby(['sector', 'type', 'n_rooms', 'n_bathroom']):
        if len(group) < 5:  # Skip small groups
            continue

        # Create area bins within each group
        group = group.copy()
        group['area_bin'] = pd.cut(group['net_usable_area'], bins=5, labels=False)

        # Detect outliers within each area bin
        for bin_id, bin_group in group.groupby('area_bin'):
            if len(bin_group) < 3:
                continue

            Q1 = bin_group['price'].quantile(0.25)
            Q3 = bin_group['price'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_mask = (bin_group['price'] < lower_bound) | (bin_group['price'] > upper_bound)
            df_clean.loc[bin_group.index[outlier_mask], 'is_outlier'] = True

    # Separate outliers
    data_without_outliers = df_clean[~df_clean['is_outlier']].drop('is_outlier', axis=1)
    data_with_outliers = df_clean[df_clean['is_outlier']].drop('is_outlier', axis=1)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Generate versioned filenames and save datasets
    clean_data_path = get_versioned_filename(
        base_name='clean',
        file_type='data',
        directory=output_dir,
        extension='csv'
    )
    outlier_data_path = get_versioned_filename(
        base_name='outliers',
        file_type='data',
        directory=output_dir,
        extension='csv'
    )

    data_without_outliers.to_csv(clean_data_path, index=False)
    data_with_outliers.to_csv(outlier_data_path, index=False)

    print(f"Data without outliers: {len(data_without_outliers)} samples")
    print(f"Data with outliers: {len(data_with_outliers)} samples")
    print(f"Files saved to:")
    print(f"  - {clean_data_path}")
    print(f"  - {outlier_data_path}")

    return data_without_outliers, data_with_outliers