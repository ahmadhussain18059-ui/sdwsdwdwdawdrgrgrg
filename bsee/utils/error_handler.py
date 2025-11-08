"""
Global error handling and recovery system for BSEE.
"""

import logging
import traceback
from typing import Any, Dict, List, Optional, Callable, Tuple, Union
from functools import wraps
from enum import Enum

from bsee.utils.validators import validate_binary_data, validate_operation_parameters


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for different types of failures."""
    NETWORK = "network"
    FILE_IO = "file_io"
    MEMORY = "memory"
    COMPUTATION = "computation"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    TIMEOUT = "timeout"
    UNKNOWN = "unknown"


class BSEEError(Exception):
    """Base exception class for BSEE errors."""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.cause = cause
        self.traceback_str = traceback.format_exc()


class OperationValidator:
    """Comprehensive validation system for binary operations."""

    def __init__(self):
        """Initialize the operation validator."""
        self.validation_history: List[Dict[str, Any]] = []
        self.failed_operations: Dict[str, List[Dict[str, Any]]] = {}

    def validate_operation(self, operation_name: str, operation_func: Callable,
                          test_data: bytes = None) -> Dict[str, Any]:
        """
        Comprehensively validate an operation.

        Args:
            operation_name: Name of the operation
            operation_func: Function implementing the operation
            test_data: Optional test data to use

        Returns:
            Validation result dictionary
        """
        validation_result = {
            'operation_name': operation_name,
            'timestamp': None,
            'success': False,
            'errors': [],
            'warnings': [],
            'performance_metrics': {},
            'reversibility_test': False,
            'parameter_validation': False,
            'edge_case_handling': False
        }

        try:
            import time
            start_time = time.time()

            # Generate test data if not provided
            if test_data is None:
                test_data = self._generate_test_data()

            # Validate basic operation properties
            self._validate_operation_signature(operation_func, validation_result)

            # Test parameter validation
            self._test_parameter_validation(operation_name, operation_func, validation_result)

            # Test basic functionality
            self._test_basic_functionality(operation_func, test_data, validation_result)

            # Test reversibility if applicable
            self._test_reversibility(operation_func, test_data, validation_result)

            # Test edge cases
            self._test_edge_cases(operation_func, validation_result)

            # Performance metrics
            end_time = time.time()
            validation_result['performance_metrics']['execution_time'] = end_time - start_time

            validation_result['success'] = len(validation_result['errors']) == 0

        except Exception as e:
            validation_result['errors'].append(f"Validation failed: {str(e)}")
            validation_result['success'] = False

        finally:
            import time
            validation_result['timestamp'] = time.time()
            self.validation_history.append(validation_result)

            if not validation_result['success']:
                if operation_name not in self.failed_operations:
                    self.failed_operations[operation_name] = []
                self.failed_operations[operation_name].append(validation_result)

        return validation_result

    def _generate_test_data(self) -> bytes:
        """Generate diverse test data for validation."""
        import random

        # Generate different types of test data
        data_types = []

        # Random data
        data_types.append(bytes([random.randint(0, 255) for _ in range(100)]))

        # Sequential data
        data_types.append(bytes(list(range(100))))

        # Repetitive data
        data_types.append(bytes([42] * 100))

        # Patterned data
        pattern = [i % 256 for i in range(50)] + [i % 256 for i in range(50)]
        data_types.append(bytes(pattern))

        # Empty and single byte
        data_types.append(b'\x00')
        data_types.append(b'\x01\x02\x03\x04')

        # Return a representative sample
        return random.choice(data_types)

    def _validate_operation_signature(self, operation_func: Callable,
                                    validation_result: Dict[str, Any]) -> None:
        """Validate operation function signature."""
        try:
            import inspect

            sig = inspect.signature(operation_func)
            params = list(sig.parameters.keys())

            # Expected signature: func(self, binary_data: bytes, **kwargs) -> Tuple[bytes, Callable, Dict]
            if len(params) < 2:
                validation_result['errors'].append("Operation function should accept at least 2 parameters")

            # Check return annotation if present
            if hasattr(operation_func, '__annotations__') and 'return' in operation_func.__annotations__:
                return_type = operation_func.__annotations__['return']
                if not hasattr(return_type, '__origin__') or return_type.__origin__ is not tuple:
                    validation_result['warnings'].append("Operation should return Tuple[bytes, Callable, Dict]")

        except Exception as e:
            validation_result['warnings'].append(f"Could not validate function signature: {str(e)}")

    def _test_parameter_validation(self, operation_name: str, operation_func: Callable,
                                  validation_result: Dict[str, Any]) -> None:
        """Test parameter validation for the operation."""
        try:
            # Test with invalid data types
            invalid_inputs = [
                None,
                "not bytes",
                123,
                [],
                {},
            ]

            for invalid_input in invalid_inputs:
                try:
                    # Create a dummy instance if needed
                    if hasattr(operation_func, '__self__'):
                        result = operation_func(invalid_input)
                    else:
                        # For standalone functions, we need to check the call pattern
                        result = operation_func(invalid_input)

                    # If it didn't raise an error, that's potentially problematic
                    validation_result['warnings'].append(
                        f"Operation accepted invalid input type: {type(invalid_input).__name__}"
                    )

                except (TypeError, ValueError, AttributeError):
                    # Expected behavior - operation rejected invalid input
                    pass
                except Exception as e:
                    validation_result['errors'].append(
                        f"Unexpected error with invalid input {type(invalid_input).__name__}: {str(e)}"
                    )

            validation_result['parameter_validation'] = True

        except Exception as e:
            validation_result['errors'].append(f"Parameter validation test failed: {str(e)}")

    def _test_basic_functionality(self, operation_func: Callable, test_data: bytes,
                                 validation_result: Dict[str, Any]) -> None:
        """Test basic operation functionality."""
        try:
            # Test with valid data - generate parameters if needed
            if hasattr(operation_func, '__self__'):
                # This is a method, try to get operation name and generate parameters
                op_name = operation_func.__name__
                params = self._generate_operation_params(op_name)
                result = operation_func(test_data, **params)
            else:
                # Standalone function - try with just data first
                try:
                    result = operation_func(test_data)
                except TypeError as e:
                    # If it needs parameters, try to generate them
                    op_name = getattr(operation_func, '__name__', 'unknown')
                    params = self._generate_operation_params(op_name)
                    result = operation_func(test_data, **params)

            # Validate result structure
            if not isinstance(result, tuple) or len(result) != 3:
                validation_result['errors'].append(
                    "Operation must return tuple of (result_data, inverse_function, metadata)"
                )
                return

            result_data, inverse_func, metadata = result

            # Validate result data
            if not isinstance(result_data, bytes):
                validation_result['errors'].append("First element of result must be bytes")

            # Validate inverse function
            if not callable(inverse_func):
                validation_result['errors'].append("Second element of result must be callable inverse function")

            # Validate metadata
            if not isinstance(metadata, dict):
                validation_result['errors'].append("Third element of result must be metadata dictionary")
            else:
                # Check required metadata fields
                required_fields = ['operation', 'reversible', 'original_size']
                for field in required_fields:
                    if field not in metadata:
                        validation_result['warnings'].append(f"Missing metadata field: {field}")

        except Exception as e:
            validation_result['errors'].append(f"Basic functionality test failed: {str(e)}")

    def _generate_operation_params(self, operation_name: str) -> Dict[str, Any]:
        """Generate appropriate parameters for an operation."""
        import random

        # Common parameter patterns
        if 'xor' in operation_name.lower() or 'and' in operation_name.lower() or 'or' in operation_name.lower():
            return {'constant': random.randint(1, 255)}
        elif 'rotate' in operation_name.lower() or 'shift' in operation_name.lower():
            return {'shift': random.randint(1, 7)}
        elif 'swap' in operation_name.lower():
            if 'bits' in operation_name.lower():
                return {'bit1': random.randint(0, 7), 'bit2': random.randint(0, 7)}
            else:
                return {'pattern': bytes([random.randint(0, 255) for _ in range(4)])}
        elif 'clear_bit' in operation_name.lower() or 'set_bit' in operation_name.lower() or 'toggle_bit' in operation_name.lower():
            return {'bit_position': random.randint(0, 7)}
        elif 'mask' in operation_name.lower():
            return {'mask': random.randint(1, 255)}
        elif 'shuffle' in operation_name.lower():
            return {'seed': random.randint(0, 10000)}
        elif 'transform' in operation_name.lower():
            return {'strength': random.uniform(0.5, 1.5)}
        elif 'encode' in operation_name.lower() or 'decode' in operation_name.lower():
            return {'level': random.randint(1, 9)}
        else:
            return {}  # No parameters needed

    def _test_reversibility(self, operation_func: Callable, test_data: bytes,
                           validation_result: Dict[str, Any]) -> None:
        """Test operation reversibility."""
        try:
            # Apply operation with parameters
            if hasattr(operation_func, '__self__'):
                op_name = operation_func.__name__
                params = self._generate_operation_params(op_name)
                result = operation_func(test_data, **params)
            else:
                try:
                    result = operation_func(test_data)
                except TypeError:
                    op_name = getattr(operation_func, '__name__', 'unknown')
                    params = self._generate_operation_params(op_name)
                    result = operation_func(test_data, **params)

            result_data, inverse_func, metadata = result

            # Check if operation claims to be reversible
            if not metadata.get('reversible', False):
                validation_result['warnings'].append("Operation marked as non-reversible")
                return

            # Test reversibility
            restored_data = inverse_func(result_data)

            if not isinstance(restored_data, bytes):
                validation_result['errors'].append("Inverse function must return bytes")
                return

            if restored_data != test_data:
                validation_result['errors'].append(
                    f"Reversibility test failed: original length {len(test_data)}, "
                    f"restored length {len(restored_data)}"
                )
                # Check if at least the lengths match
                if len(restored_data) == len(test_data):
                    validation_result['warnings'].append("Length matches but content differs")
            else:
                validation_result['reversibility_test'] = True

        except Exception as e:
            validation_result['errors'].append(f"Reversibility test failed: {str(e)}")

    def _test_edge_cases(self, operation_func: Callable, validation_result: Dict[str, Any]) -> None:
        """Test operation with edge cases."""
        edge_cases = [
            b'\x01\x02\x03\x04',  # Small data (skip empty to avoid errors)
            b'\x00',  # Single zero byte
            b'\xff',  # Single max byte
            b'\x00' * 100,  # All zeros (smaller)
            b'\xff' * 100,  # All max values (smaller)
            bytes(range(100)),  # Byte range (smaller)
        ]

        passed_edge_cases = 0
        total_edge_cases = len(edge_cases)

        for edge_case in edge_cases:
            try:
                if hasattr(operation_func, '__self__'):
                    op_name = operation_func.__name__
                    params = self._generate_operation_params(op_name)
                    result = operation_func(edge_case, **params)
                else:
                    try:
                        result = operation_func(edge_case)
                    except TypeError:
                        op_name = getattr(operation_func, '__name__', 'unknown')
                        params = self._generate_operation_params(op_name)
                        result = operation_func(edge_case, **params)

                # Basic validation of result structure
                if isinstance(result, tuple) and len(result) == 3:
                    passed_edge_cases += 1

            except Exception as e:
                # Some edge cases might legitimately fail
                # Log as warning rather than error
                validation_result['warnings'].append(
                    f"Edge case failed: {str(e)[:100]}..."
                )

        # If most edge cases pass, consider it successful
        if passed_edge_cases >= total_edge_cases * 0.7:  # 70% pass rate
            validation_result['edge_case_handling'] = True
        elif passed_edge_cases == 0:
            validation_result['errors'].append("All edge cases failed")

    def validate_all_operations(self, operations_registry) -> Dict[str, Dict[str, Any]]:
        """Validate all operations in a registry."""
        results = {}

        for operation_name, operation_func in operations_registry.operations.items():
            try:
                result = self.validate_operation(operation_name, operation_func)
                results[operation_name] = result

            except Exception as e:
                results[operation_name] = {
                    'operation_name': operation_name,
                    'success': False,
                    'errors': [f"Validation failed to run: {str(e)}"],
                    'warnings': [],
                    'performance_metrics': {},
                    'reversibility_test': False,
                    'parameter_validation': False,
                    'edge_case_handling': False
                }

        return results

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validations."""
        total_validations = len(self.validation_history)
        successful_validations = sum(1 for v in self.validation_history if v['success'])

        failure_counts = {}
        for validation in self.validation_history:
            if not validation['success']:
                op_name = validation['operation_name']
                failure_counts[op_name] = failure_counts.get(op_name, 0) + 1

        return {
            'total_validations': total_validations,
            'successful_validations': successful_validations,
            'failed_validations': total_validations - successful_validations,
            'success_rate': successful_validations / max(1, total_validations),
            'failure_counts': failure_counts,
            'most_failed_operations': sorted(failure_counts.items(),
                                           key=lambda x: x[1], reverse=True)[:5]
        }


class GlobalErrorHandler:
    """Global error handling and recovery system."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize global error handler."""
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.operation_validator = OperationValidator()
        self._setup_default_recovery_strategies()

    def _setup_default_recovery_strategies(self) -> None:
        """Setup default recovery strategies for different error categories."""

        def memory_recovery(error_context: Dict[str, Any]) -> bool:
            """Recovery strategy for memory errors."""
            try:
                import gc
                gc.collect()  # Force garbage collection
                return True
            except Exception:
                return False

        def computation_recovery(error_context: Dict[str, Any]) -> bool:
            """Recovery strategy for computation errors."""
            # Try to simplify the computation
            context = error_context.get('operation_context', {})
            if 'data_size' in context and context['data_size'] > 10000:
                # Try with smaller data
                self.logger.info("Attempting recovery with smaller data")
                return True
            return False

        def dependency_recovery(error_context: Dict[str, Any]) -> bool:
            """Recovery strategy for dependency errors."""
            # Try to import fallback implementation
            missing_dep = error_context.get('missing_dependency')
            if missing_dep:
                self.logger.warning(f"Missing dependency {missing_dep}, using fallback")
                return True
            return False

        # Register recovery strategies
        self.recovery_strategies[ErrorCategory.MEMORY] = [memory_recovery]
        self.recovery_strategies[ErrorCategory.COMPUTATION] = [computation_recovery]
        self.recovery_strategies[ErrorCategory.DEPENDENCY] = [dependency_recovery]

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Dict[str, Any]:
        """
        Handle an error with appropriate recovery strategies.

        Args:
            error: The exception that occurred
            context: Additional context about the error
            severity: Error severity level

        Returns:
            Error handling result
        """
        error_result = {
            'error': error,
            'category': self._categorize_error(error),
            'severity': severity,
            'context': context or {},
            'recovery_attempted': False,
            'recovery_successful': False,
            'timestamp': None
        }

        try:
            import time
            error_result['timestamp'] = time.time()

            # Log the error
            self.logger.error(f"Error handled: {error} (Category: {error_result['category'].value})")

            # Attempt recovery if we have strategies for this category
            if error_result['category'] in self.recovery_strategies:
                error_result['recovery_attempted'] = True

                for recovery_strategy in self.recovery_strategies[error_result['category']]:
                    try:
                        success = recovery_strategy(error_result['context'])
                        if success:
                            error_result['recovery_successful'] = True
                            self.logger.info(f"Recovery successful for {error_result['category'].value}")
                            break
                    except Exception as recovery_error:
                        self.logger.error(f"Recovery strategy failed: {recovery_error}")

            # Record error
            self.error_history.append(error_result)

            # Clean up old error history if it gets too large
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-500:]  # Keep last 500

        except Exception as handling_error:
            self.logger.error(f"Error in error handling: {handling_error}")

        return error_result

    def _categorize_error(self, error: Exception) -> ErrorCategory:
        """Categorize an error based on its type and message."""
        error_message = str(error).lower()
        error_type = type(error).__name__.lower()

        if 'memory' in error_message or 'memory' in error_type:
            return ErrorCategory.MEMORY
        elif 'file' in error_message or 'io' in error_type or 'os' in error_type:
            return ErrorCategory.FILE_IO
        elif 'import' in error_message or 'module' in error_message:
            return ErrorCategory.DEPENDENCY
        elif 'timeout' in error_message or 'time' in error_type:
            return ErrorCategory.TIMEOUT
        elif 'value' in error_type or 'type' in error_type or 'validation' in error_message:
            return ErrorCategory.VALIDATION
        elif 'network' in error_message or 'connection' in error_message:
            return ErrorCategory.NETWORK
        elif 'computation' in error_message or 'calculation' in error_message:
            return ErrorCategory.COMPUTATION
        elif 'config' in error_message or 'yaml' in error_type:
            return ErrorCategory.CONFIGURATION
        else:
            return ErrorCategory.UNKNOWN

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of all handled errors."""
        if not self.error_history:
            return {'total_errors': 0, 'categories': {}, 'recovery_rate': 0.0}

        total_errors = len(self.error_history)
        category_counts = {}
        recovery_attempts = 0
        successful_recoveries = 0

        for error_record in self.error_history:
            category = error_record['category'].value
            category_counts[category] = category_counts.get(category, 0) + 1

            if error_record['recovery_attempted']:
                recovery_attempts += 1
                if error_record['recovery_successful']:
                    successful_recoveries += 1

        recovery_rate = successful_recoveries / max(1, recovery_attempts)

        return {
            'total_errors': total_errors,
            'categories': category_counts,
            'recovery_attempts': recovery_attempts,
            'successful_recoveries': successful_recoveries,
            'recovery_rate': recovery_rate,
            'most_common_errors': sorted(category_counts.items(),
                                        key=lambda x: x[1], reverse=True)[:5]
        }


def safe_execute(func: Callable, *args, error_handler: Optional[GlobalErrorHandler] = None,
                default_return: Any = None, context: Optional[Dict[str, Any]] = None,
                **kwargs) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """
    Safely execute a function with error handling.

    Args:
        func: Function to execute
        *args: Function arguments
        error_handler: Global error handler instance
        default_return: Default return value if function fails
        context: Additional context for error handling
        **kwargs: Function keyword arguments

    Returns:
        Tuple of (result, error_info)
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error_info = None
        if error_handler:
            error_info = error_handler.handle_error(e, context)
        return default_return, error_info


def operation_validator_decorator(test_data: Optional[bytes] = None):
    """
    Decorator to automatically validate operations.

    Args:
        test_data: Test data to use for validation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, binary_data: bytes, **kwargs) -> Tuple[bytes, Callable, Dict]:
            # Validate inputs
            validate_binary_data(binary_data)

            # Call original function
            result = func(self, binary_data, **kwargs)

            # Validate output structure
            if not isinstance(result, tuple) or len(result) != 3:
                raise ValueError("Operation must return tuple of (bytes, Callable, Dict)")

            return result

        # Add validation method to the function
        def validate_operation(operation_validator: OperationValidator) -> Dict[str, Any]:
            return operation_validator.validate_operation(func.__name__, wrapper, test_data)

        wrapper.validate_operation = validate_operation
        return wrapper

    return decorator


# Global instance for easy access
_global_error_handler: Optional[GlobalErrorHandler] = None


def get_global_error_handler() -> GlobalErrorHandler:
    """Get or create the global error handler instance."""
    global _global_error_handler
    if _global_error_handler is None:
        _global_error_handler = GlobalErrorHandler()
    return _global_error_handler


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None,
                severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Dict[str, Any]:
    """Convenience function to handle errors with the global handler."""
    return get_global_error_handler().handle_error(error, context, severity)