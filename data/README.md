# Data Directory

This directory is for storing the datasets used by the Property-Friends Real Estate Valuation system.

## Required Files

The following files should be placed in this directory:

1. `train.csv` - Training dataset used for model training
2. `test.csv` - Test dataset used for model evaluation

## Important Note

These files are not included in the repository as they contain proprietary data. You will need to obtain these files separately and place them in this directory before running the training pipeline.

## File Format

Both files should be in CSV format with the following columns:
- type: Property type (e.g., departamento, casa)
- sector: Neighborhood or area
- net_usable_area: Usable area in square meters
- net_area: Total area in square meters
- n_rooms: Number of rooms
- n_bathroom: Number of bathrooms
- latitude: Geographic latitude
- longitude: Geographic longitude
- price: Property price (target variable)