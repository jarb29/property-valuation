"""
ORM-style data access layer for property valuation data.

This module provides an ORM-like interface for data access with support
for both file-based and database storage backends.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List, Union
from datetime import datetime
import pandas as pd


class BaseModel(ABC):
    """Base ORM-like model class."""
    
    def __init__(self, data_source: 'DataSource'):
        self._data_source = data_source
        self._data: Optional[pd.DataFrame] = None
    
    @property
    @abstractmethod
    def table_name(self) -> str:
        """Table/file identifier for this model."""
        pass
    
    def all(self, version: Optional[str] = None) -> pd.DataFrame:
        """Get all records."""
        return self._data_source.load_data(self.table_name, version)
    
    def filter(self, **kwargs) -> pd.DataFrame:
        """Filter records by conditions."""
        data = self.all()
        for key, value in kwargs.items():
            if key in data.columns:
                data = data[data[key] == value]
        return data
    
    def create(self, data: pd.DataFrame, version: str) -> str:
        """Create new records."""
        return self._data_source.save_data(data, self.table_name, version)
    
    def get_latest(self) -> pd.DataFrame:
        """Get latest version of data."""
        latest_version = self._data_source.get_latest_version(self.table_name)
        return self.all(latest_version) if latest_version else pd.DataFrame()
    
    def versions(self) -> List[str]:
        """List all available versions."""
        return self._data_source.list_versions(self.table_name)


class PropertyTrainData(BaseModel):
    """ORM model for training data."""
    
    @property
    def table_name(self) -> str:
        return "train"
    
    def by_property_type(self, property_type: str) -> pd.DataFrame:
        """Filter by property type."""
        return self.filter(type=property_type)
    
    def by_sector(self, sector: str) -> pd.DataFrame:
        """Filter by sector."""
        return self.filter(sector=sector)
    
    def price_range(self, min_price: float, max_price: float) -> pd.DataFrame:
        """Filter by price range."""
        data = self.all()
        return data[(data['price'] >= min_price) & (data['price'] <= max_price)]
    
    def area_range(self, min_area: float, max_area: float) -> pd.DataFrame:
        """Filter by area range."""
        data = self.all()
        return data[(data['net_usable_area'] >= min_area) & (data['net_usable_area'] <= max_area)]


class PropertyTestData(BaseModel):
    """ORM model for test data."""
    
    @property
    def table_name(self) -> str:
        return "test"
    
    def by_property_type(self, property_type: str) -> pd.DataFrame:
        """Filter by property type."""
        return self.filter(type=property_type)
    
    def by_sector(self, sector: str) -> pd.DataFrame:
        """Filter by sector."""
        return self.filter(sector=sector)


class ProcessedData(BaseModel):
    """ORM model for processed data."""
    
    def __init__(self, data_source: 'DataSource', data_type: str):
        super().__init__(data_source)
        self._data_type = data_type
    
    @property
    def table_name(self) -> str:
        return f"processed_{self._data_type}"
    
    def clean_data(self) -> pd.DataFrame:
        """Get clean data (no zeros)."""
        data = self.all()
        return data[~(data == 0).any(axis=1)]
    
    def outliers_removed(self) -> pd.DataFrame:
        """Get data with outliers removed."""
        return self.filter(outlier_removed=True)


class DataSource(ABC):
    """Abstract data source for ORM models."""
    
    @abstractmethod
    def load_data(self, table_name: str, version: Optional[str] = None, **kwargs) -> pd.DataFrame:
        pass
    
    @abstractmethod
    def save_data(self, data: pd.DataFrame, table_name: str, version: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def list_versions(self, table_name: str) -> List[str]:
        pass
    
    @abstractmethod
    def get_latest_version(self, table_name: str) -> Optional[str]:
        pass


class FileDataSource(DataSource):
    """File-based data source for ORM."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def load_data(self, table_name: str, version: Optional[str] = None, **kwargs) -> pd.DataFrame:
        # Implementation would use existing file loading logic
        pass
    
    def save_data(self, data: pd.DataFrame, table_name: str, version: str, **kwargs) -> str:
        # Implementation would use existing file saving logic
        pass
    
    def list_versions(self, table_name: str) -> List[str]:
        pass
    
    def get_latest_version(self, table_name: str) -> Optional[str]:
        pass


class DatabaseDataSource(DataSource):
    """Database-based data source for ORM."""
    
    def __init__(self, connection_string: str, table_prefix: str = "property_"):
        self.connection_string = connection_string
        self.table_prefix = table_prefix
    
    def load_data(self, table_name: str, version: Optional[str] = None, **kwargs) -> pd.DataFrame:
        # Future implementation: SQL queries with pandas
        pass
    
    def save_data(self, data: pd.DataFrame, table_name: str, version: str, **kwargs) -> str:
        # Future implementation: DataFrame.to_sql()
        pass
    
    def list_versions(self, table_name: str) -> List[str]:
        pass
    
    def get_latest_version(self, table_name: str) -> Optional[str]:
        pass


class PropertyORM:
    """Main ORM interface for property data."""
    
    def __init__(self, data_source: DataSource):
        self.data_source = data_source
        self.train = PropertyTrainData(data_source)
        self.test = PropertyTestData(data_source)
    
    def processed(self, data_type: str) -> ProcessedData:
        """Get processed data model."""
        return ProcessedData(self.data_source, data_type)
    
    def create_version(self, version: str) -> 'VersionContext':
        """Create a version context for batch operations."""
        return VersionContext(self, version)


class VersionContext:
    """Context manager for version-specific operations."""
    
    def __init__(self, orm: PropertyORM, version: str):
        self.orm = orm
        self.version = version
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def save_train_data(self, data: pd.DataFrame) -> str:
        """Save training data for this version."""
        return self.orm.train.create(data, self.version)
    
    def save_test_data(self, data: pd.DataFrame) -> str:
        """Save test data for this version."""
        return self.orm.test.create(data, self.version)


# Usage examples:
"""
# Initialize ORM
file_source = FileDataSource("/path/to/data")
db_source = DatabaseDataSource("postgresql://user:pass@host/db")

# File-based ORM
orm = PropertyORM(file_source)

# Database ORM  
orm = PropertyORM(db_source)

# Query operations
train_data = orm.train.all()
apartments = orm.train.by_property_type("departamento")
las_condes = orm.train.by_sector("las condes")
expensive = orm.train.price_range(100000000, 500000000)

# Version operations
latest_train = orm.train.get_latest()
versions = orm.train.versions()

# Batch operations with version context
with orm.create_version("v3.5") as version:
    version.save_train_data(processed_train_data)
    version.save_test_data(processed_test_data)

# Processed data
clean_data = orm.processed("train").clean_data()
"""