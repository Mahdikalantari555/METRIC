"""
Unit tests for exceptions module of METRIC ETa pipeline.

Tests custom exception hierarchy and utilities.
"""

import pytest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestMETRICError:
    """Test base METRIC exception."""
    
    def test_error_creation(self):
        """Test METRICError creation."""
        from metric_et.utils.exceptions import METRICError
        
        error = METRICError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
    
    def test_error_with_details(self):
        """Test METRICError with details."""
        from metric_et.utils.exceptions import METRICError
        
        error = METRICError("Test error", details={"key": "value"})
        
        assert error.details["key"] == "value"
        assert "value" in str(error)
    
    def test_error_add_detail(self):
        """Test adding details to error."""
        from metric_et.utils.exceptions import METRICError
        
        error = METRICError("Test error")
        error.add_detail("stage", "calibration")
        
        assert error.details["stage"] == "calibration"


class TestDataInputError:
    """Test DataInputError and subclasses."""
    
    def test_data_input_error(self):
        """Test DataInputError creation."""
        from metric_et.utils.exceptions import DataInputError
        
        error = DataInputError("Invalid data", input_type="weather", file_path="/data.csv")
        
        assert error.details["input_type"] == "weather"
        assert error.details["file_path"] == "/data.csv"
    
    def test_scene_path_error(self):
        """Test ScenePathError creation."""
        from metric_et.utils.exceptions import ScenePathError
        
        error = ScenePathError("Invalid scene path", scene_path="/path/to/scene")
        
        assert error.details["input_type"] == "landsat_scene"
        assert error.details["file_path"] == "/path/to/scene"


class TestCalibrationError:
    """Test CalibrationError and subclasses."""
    
    def test_calibration_error(self):
        """Test CalibrationError creation."""
        from metric_et.utils.exceptions import CalibrationError
        
        error = CalibrationError("Calibration failed", stage="anchor_selection")
        
        assert error.details["stage"] == "anchor_selection"


class TestComputationError:
    """Test ComputationError and subclasses."""
    
    def test_computation_error(self):
        """Test ComputationError creation."""
        from metric_et.utils.exceptions import ComputationError
        
        error = ComputationError("Numerical overflow", computation_step="radiation")
        
        assert error.details["step"] == "radiation"


class TestOutputError:
    """Test OutputError and subclasses."""
    
    def test_output_error(self):
        """Test OutputError creation."""
        from metric_et.utils.exceptions import OutputError
        
        error = OutputError("Write failed", output_path="/output.tif", output_type="geotiff")
        
        assert error.details["output_path"] == "/output.tif"
        assert error.details["output_type"] == "geotiff"


class TestErrorHandlingUtilities:
    """Test error handling utilities."""
    
    def test_handle_exception_decorator(self):
        """Test handle_exception decorator."""
        from metric_et.utils.exceptions import handle_exception, ComputationError
        
        @handle_exception
        def failing_function():
            raise ValueError("Test value error")
        
        # Should convert to ComputationError
        with pytest.raises(ComputationError):
            failing_function()
    
    def test_handle_exception_file_not_found(self):
        """Test handle_exception with FileNotFoundError."""
        from metric_et.utils.exceptions import handle_exception, DataInputError
        
        @handle_exception
        def file_operation():
            raise FileNotFoundError("test.txt")
        
        with pytest.raises(DataInputError):
            file_operation()
    
    def test_create_error_context(self):
        """Test create_error_context function."""
        from metric_et.utils.exceptions import METRICError, create_error_context
        
        error = METRICError("Test error", details={"code": 123})
        context = create_error_context(error, {"user": "test_user"})
        
        assert context["error_type"] == "METRICError"
        assert context["error_details"]["code"] == 123
        assert context["additional_context"]["user"] == "test_user"


class TestExceptionHierarchy:
    """Test exception hierarchy."""
    
    def test_exception_inheritance(self):
        """Test METRIC exception inheritance."""
        from metric_et.utils.exceptions import (
            METRICError,
            DataInputError,
            CalibrationError,
            ComputationError,
            OutputError
        )
        
        # All should inherit from METRICError
        assert issubclass(DataInputError, METRICError)
        assert issubclass(CalibrationError, METRICError)
        assert issubclass(ComputationError, METRICError)
        assert issubclass(OutputError, METRICError)
    
    def test_exception_catching(self):
        """Test catching exceptions by base class."""
        from metric_et.utils.exceptions import METRICError, DataInputError
        
        try:
            raise DataInputError("Test")
        except METRICError as e:
            assert isinstance(e, DataInputError)
            assert str(e) == "Test"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
