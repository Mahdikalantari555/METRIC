"""Custom exceptions for I/O operations in METRIC ETa model."""

from typing import List, Optional


class LandsatIOError(Exception):
    """Base exception for Landsat I/O errors."""
    pass


class NoSceneFoundError(LandsatIOError):
    """Raised when no suitable Landsat scene is found."""
    
    def __init__(self, message: str, partial_count: Optional[int] = None):
        super().__init__(message)
        self.partial_count = partial_count
        self.message = message
    
    def __str__(self):
        if self.partial_count is not None:
            return f"{self.message} (partial overlaps: {self.partial_count})"
        return self.message


class AuthenticationError(LandsatIOError):
    """Raised when authentication with Planetary Computer fails."""
    pass


class DownloadError(LandsatIOError):
    """Raised when band download fails."""
    
    def __init__(self, message: str, band_name: Optional[str] = None, url: Optional[str] = None):
        super().__init__(message)
        self.band_name = band_name
        self.url = url
        self.message = message
    
    def __str__(self):
        details = []
        if self.band_name:
            details.append(f"band: {self.band_name}")
        if self.url:
            details.append(f"url: {self.url}")
        if details:
            return f"{self.message} ({', '.join(details)})"
        return self.message


class PartialDataError(LandsatIOError):
    """Raised when some required bands are missing."""
    
    def __init__(self, message: str, missing_bands: List[str], available_bands: List[str]):
        super().__init__(message)
        self.missing_bands = missing_bands
        self.available_bands = available_bands
        self.message = message
    
    def __str__(self):
        missing_str = ", ".join(self.missing_bands)
        return f"{self.message} Missing bands: [{missing_str}]"


class GeometryError(LandsatIOError):
    """Raised when geometry processing fails."""
    pass
