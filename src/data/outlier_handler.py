import pandas as pd
import os

from src.config import PIPELINE_DATA_DIR, JUPYTER_DATA_DIR, TRAIN_DATA_PATH
from src.utils.helpers import get_versioned_filename


def separate_outliers_and_save(data_path=None, save_target='pipeline'):
    output_dir = PIPELINE_DATA_DIR if save_target == 'pipeline' else JUPYTER_DATA_DIR
    data_path = data_path or TRAIN_DATA_PATH
    
    df = pd.read_csv(data_path)
    df_clean = df[~(df == 0).any(axis=1)].copy()
    df_clean['is_outlier'] = False

    for (sector, prop_type, rooms, bathrooms), group in df_clean.groupby(['sector', 'type', 'n_rooms', 'n_bathroom']):
        if len(group) < 5:
            continue

        group = group.copy()
        group['area_bin'] = pd.cut(group['net_usable_area'], bins=5, labels=False)

        for bin_id, bin_group in group.groupby('area_bin'):
            if len(bin_group) < 3:
                continue

            Q1, Q3 = bin_group['price'].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR

            outlier_mask = (bin_group['price'] < lower_bound) | (bin_group['price'] > upper_bound)
            df_clean.loc[bin_group.index[outlier_mask], 'is_outlier'] = True

    data_without_outliers = df_clean[~df_clean['is_outlier']].drop('is_outlier', axis=1)
    data_with_outliers = df_clean[df_clean['is_outlier']].drop('is_outlier', axis=1)

    os.makedirs(output_dir, exist_ok=True)

    clean_data_path = get_versioned_filename('clean', 'data', output_dir, 'csv')
    outlier_data_path = get_versioned_filename('outliers', 'data', output_dir, 'csv')

    data_without_outliers.to_csv(clean_data_path, index=False)
    data_with_outliers.to_csv(outlier_data_path, index=False)

    return data_without_outliers, data_with_outliers