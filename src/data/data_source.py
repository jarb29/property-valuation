"""
Data source abstraction layer for future database integration.

This module provides an abstract interface that can be easily adapted
to work with databases while maintaining compatibility with the current
file-based implementation.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
import pandas as pd


class DataSource(ABC):
    """Abstract base class for data sources."""
    
    @abstractmethod
    def load_data(self, data_type: str, version: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from the source."""
        pass
    
    @abstractmethod
    def save_data(self, data: pd.DataFrame, data_type: str, version: str, **kwargs) -> str:
        """Save data to the source."""
        pass
    
    @abstractmethod
    def list_versions(self, data_type: str) -> List[str]:
        """List available versions for a data type."""
        pass
    
    @abstractmethod
    def get_latest_version(self, data_type: str) -> Optional[str]:
        """Get the latest version for a data type."""
        pass


class FileDataSource(DataSource):
    """File-based data source implementation."""
    
    def __init__(self, base_path: str):
        self.base_path = base_path
    
    def load_data(self, data_type: str, version: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from CSV files."""
        # Implementation would use existing file loading logic
        pass
    
    def save_data(self, data: pd.DataFrame, data_type: str, version: str, **kwargs) -> str:
        """Save data to CSV files."""
        # Implementation would use existing file saving logic
        pass
    
    def list_versions(self, data_type: str) -> List[str]:
        """List available file versions."""
        pass
    
    def get_latest_version(self, data_type: str) -> Optional[str]:
        """Get latest file version."""
        pass


class DatabaseDataSource(DataSource):
    """Database-based data source implementation (future)."""
    
    def __init__(self, connection_string: str, table_prefix: str = "property_"):
        self.connection_string = connection_string
        self.table_prefix = table_prefix
    
    def load_data(self, data_type: str, version: Optional[str] = None, **kwargs) -> pd.DataFrame:
        """Load data from database tables."""
        # Future implementation:
        # - Connect to database
        # - Query data with version filtering
        # - Return as DataFrame
        pass
    
    def save_data(self, data: pd.DataFrame, data_type: str, version: str, **kwargs) -> str:
        """Save data to database tables."""
        # Future implementation:
        # - Connect to database
        # - Insert/update data with version
        # - Return table/version identifier
        pass
    
    def list_versions(self, data_type: str) -> List[str]:
        """List available versions from database."""
        # Future implementation:
        # - Query version metadata table
        # - Return sorted list of versions
        pass
    
    def get_latest_version(self, data_type: str) -> Optional[str]:
        """Get latest version from database."""
        # Future implementation:
        # - Query max version from metadata
        # - Return latest version string
        pass


class DataRepository:
    """Repository pattern for data access with pluggable sources."""
    
    def __init__(self, data_source: DataSource):
        self.data_source = data_source
    
    def get_train_data(self, version: Optional[str] = None) -> pd.DataFrame:
        """Get training data."""
        return self.data_source.load_data("train", version)
    
    def get_test_data(self, version: Optional[str] = None) -> pd.DataFrame:
        """Get test data."""
        return self.data_source.load_data("test", version)
    
    def save_processed_data(self, data: pd.DataFrame, data_type: str, version: str) -> str:
        """Save processed data."""
        return self.data_source.save_data(data, data_type, version)
    
    def get_latest_data(self, data_type: str) -> pd.DataFrame:
        """Get latest version of data."""
        latest_version = self.data_source.get_latest_version(data_type)
        return self.data_source.load_data(data_type, latest_version)
    

#     # Current (file-based)
# file_source = FileDataSource("/path/to/data")
# repo = DataRepository(file_source)

# # Future (database)
# db_source = DatabaseDataSource("postgresql://user:pass@host/db")
# repo = DataRepository(db_source)

# # Same interface for both
# train_data = repo.get_train_data()
