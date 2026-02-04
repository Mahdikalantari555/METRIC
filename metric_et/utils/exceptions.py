"""
Custom exceptions for METRIC ETa pipeline.

Provides a hierarchical exception system for better error handling
and debugging.
"""


class METRICError(Exception):
    """
    Base exception for METRIC errors.
    
    All custom METRIC exceptions inherit from this class.
    Provides context information about the error location and details.
    """
    
    def __init__(self, message: str, details: dict = None, *args):
        super().__init__(message, *args)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message
    
    def add_detail(self, key: str, value) -> None:
        """
        Add detail information to the exception.
        
        Args:
            key: Detail key
            value: Detail value
        """
        self.details[key] = value


class DataInputError(METRICError):
    """
    Exception raised for invalid input data.
    
    This includes:
    - Missing or corrupted input files
    - Invalid data format
    - Data validation failures
    - Coordinate reference system mismatches
    """
    
    def __init__(self, message: str, input_type: str = None, file_path: str = None, *args):
        details = {}
        if input_type:
            details["input_type"] = input_type
        if file_path:
            details["file_path"] = file_path
        super().__init__(message, details, *args)


class ScenePathError(DataInputError):
    """
    Exception raised for invalid Landsat scene paths.
    """
    
    def __init__(self, message: str, scene_path: str = None, *args):
        super().__init__(message, input_type="landsat_scene", file_path=scene_path, *args)


class WeatherDataError(DataInputError):
    """
    Exception raised for invalid weather data.
    """
    
    def __init__(self, message: str, missing_columns: list = None, *args):
        details = {}
        if missing_columns:
            details["missing_columns"] = missing_columns
        super().__init__(message, input_type="weather_data", details=details, *args)


class CalibrationError(METRICError):
    """
    Exception raised for calibration failures.
    
    This includes:
    - Invalid anchor pixel selection
    - Calibration coefficient computation errors
    - dT calibration failures
    """
    
    def __init__(self, message: str, stage: str = None, anchor_pixel: str = None, *args):
        details = {}
        if stage:
            details["stage"] = stage
        if anchor_pixel:
            details["anchor_pixel"] = anchor_pixel
        super().__init__(message, details, *args)


class AnchorPixelError(CalibrationError):
    """
    Exception raised for anchor pixel issues.
    """
    
    def __init__(self, message: str, pixel_type: str = None, coordinates: tuple = None, *args):
        details = {}
        if pixel_type:
            details["pixel_type"] = pixel_type
        if coordinates:
            details["coordinates"] = coordinates
        super().__init__(message, stage="anchor_pixel_selection", details=details, *args)


class DTCalibrationError(CalibrationError):
    """
    Exception raised for dT calibration failures.
    """
    
    def __init__(self, message: str, Ts_cold: float = None, Ts_hot: float = None, *args):
        details = {}
        if Ts_cold is not None:
            details["Ts_cold"] = Ts_cold
        if Ts_hot is not None:
            details["Ts_hot"] = Ts_hot
        super().__init__(message, stage="dT_calibration", details=details, *args)


class ComputationError(METRICError):
    """
    Exception raised for computational errors in energy balance.
    
    This includes:
    - Numerical overflow/underflow
    - Invalid intermediate calculations
    - Array dimension mismatches
    """
    
    def __init__(self, message: str, computation_step: str = None, *args):
        details = {}
        if computation_step:
            details["step"] = computation_step
        super().__init__(message, details, *args)


class RadiationBalanceError(ComputationError):
    """
    Exception raised for radiation balance calculation errors.
    """
    
    def __init__(self, message: str, component: str = None, *args):
        details = {}
        if component:
            details["component"] = component
        super().__init__(message, computation_step="radiation_balance", details=details, *args)


class EnergyBalanceError(ComputationError):
    """
    Exception raised for energy balance calculation errors.
    """
    
    def __init__(self, message: str, component: str = None, *args):
        details = {}
        if component:
            details["component"] = component
        super().__init__(message, computation_step="energy_balance", details=details, *args)


class OutputError(METRICError):
    """
    Exception raised for output file errors.
    
    This includes:
    - File write failures
    - Invalid output format
    - Disk space issues
    """
    
    def __init__(self, message: str, output_path: str = None, output_type: str = None, *args):
        details = {}
        if output_path:
            details["output_path"] = output_path
        if output_type:
            details["output_type"] = output_type
        super().__init__(message, details, *args)


class GeoTIFFWriteError(OutputError):
    """
    Exception raised for GeoTIFF write failures.
    """
    
    def __init__(self, message: str, file_path: str = None, *args):
        super().__init__(message, output_type="geotiff", file_path=file_path, *args)


class VisualizationError(OutputError):
    """
    Exception raised for visualization generation failures.
    """
    
    def __init__(self, message: str, plot_type: str = None, *args):
        details = {}
        if plot_type:
            details["plot_type"] = plot_type
        super().__init__(message, output_type="visualization", details=details, *args)


class ConfigurationError(METRICError):
    """
    Exception raised for configuration errors.
    
    This includes:
    - Missing configuration parameters
    - Invalid configuration values
    - Environment variable issues
    """
    
    def __init__(self, message: str, config_param: str = None, *args):
        details = {}
        if config_param:
            details["parameter"] = config_param
        super().__init__(message, details, *args)


class PipelineError(METRICError):
    """
    Exception raised for pipeline execution errors.
    
    This includes:
    - Pipeline stage failures
    - Step dependency issues
    - Resource allocation problems
    """
    
    def __init__(self, message: str, pipeline_stage: str = None, step: str = None, *args):
        details = {}
        if pipeline_stage:
            details["stage"] = pipeline_stage
        if step:
            details["step"] = step
        super().__init__(message, details, *args)


# Error handling utilities

def handle_exception(func):
    """
    Decorator to wrap functions with exception handling.
    
    Converts exceptions to METRIC-specific exceptions while preserving
    the original error context.
    
    Usage:
        @handle_exception
        def my_function():
            pass
    """
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except METRICError:
            # Re-raise METRIC errors as-is
            raise
        except ValueError as e:
            raise ComputationError(
                f"Value error in {func.__name__}: {e}",
                computation_step=func.__name__
            ) from e
        except FileNotFoundError as e:
            raise DataInputError(
                f"File not found: {e}",
                file_path=str(e.filename)
            ) from e
        except PermissionError as e:
            raise OutputError(
                f"Permission denied: {e}",
                output_path=str(e.filename)
            ) from e
        except Exception as e:
            raise METRICError(
                f"Unexpected error in {func.__name__}: {e}"
            ) from e
    return wrapper


def create_error_context(error: Exception, context: dict) -> dict:
    """
    Create a comprehensive error context dictionary.
    
    Args:
        error: The exception that occurred
        context: Additional context information
        
    Returns:
        Dictionary with error details
    """
    context_data = {
        "error_type": type(error).__name__,
        "error_message": str(error),
    }
    
    if hasattr(error, 'details'):
        context_data["error_details"] = error.details
    
    if context:
        context_data["additional_context"] = context
    
    return context_data
