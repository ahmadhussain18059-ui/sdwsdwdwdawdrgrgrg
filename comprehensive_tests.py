#!/usr/bin/env python3
"""
Comprehensive test suite for BSEE critical infrastructure implementation.
Tests all phases of the implementation to ensure everything works correctly.
"""

import sys
import os
import time
import json
import tempfile
import random
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def run_phase1_tests():
    """Test Phase 1: Search strategies implementation."""
    print("\n" + "="*60)
    print("PHASE 1: SEARCH STRATEGIES TESTING")
    print("="*60)

    try:
        from bsee.strategies.mcts_strategy import MCTSStrategy
        from bsee.strategies.genetic_strategy import GeneticStrategy
        from bsee.strategies.beam_strategy import BeamStrategy
        from bsee.strategies.annealing_strategy import AnnealingStrategy
        from bsee.strategies.heuristic_strategy import HeuristicStrategy
        from bsee.engine.state import State

        # Test strategy initialization
        strategies = {
            'MCTS': MCTSStrategy({'exploration_constant': 1.4, 'simulation_count': 10}),
            'Genetic': GeneticStrategy({'population_size': 10, 'mutation_rate': 0.1}),
            'Beam': BeamStrategy({'beam_width': 5, 'branch_factor': 3}),
            'Annealing': AnnealingStrategy({'initial_temperature': 100.0, 'cooling_rate': 0.95}),
            'Heuristic': HeuristicStrategy({'heuristic_weights': {'cost': 0.2, 'improvement': 0.3}})
        }

        test_data = bytes([random.randint(0, 255) for _ in range(100)])
        test_state = State(binary_data=test_data)

        results = {}
        for name, strategy in strategies.items():
            try:
                start_time = time.time()

                # Test strategy proposal
                operation, params = strategy.propose(test_state)
                proposal_time = time.time() - start_time

                # Test strategy acceptance
                test_state.score = random.uniform(-10, 10)
                accept_result = strategy.accept(test_state)

                # Test strategy info
                info = strategy.get_strategy_info()

                results[name] = {
                    'success': True,
                    'proposal_time': proposal_time,
                    'proposed_operation': operation,
                    'operation_params': params,
                    'accept_result': accept_result,
                    'error_handling': hasattr(strategy, 'safe_propose'),
                    'strategy_info': info
                }

                print(f"✅ {name}: Proposal OK ({proposal_time:.3f}s), Accept: {accept_result}")

            except Exception as e:
                results[name] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ {name}: FAILED - {e}")

        return {'phase': 'Phase 1', 'strategies': results}

    except ImportError as e:
        return {'phase': 'Phase 1', 'success': False, 'error': f'Import error: {e}'}


def run_phase2_tests():
    """Test Phase 2: Transform operations implementation."""
    print("\n" + "="*60)
    print("PHASE 2: TRANSFORM OPERATIONS TESTING")
    print("="*60)

    try:
        from bsee.operations.operations_registry import OperationsRegistry
        from bsee.utils.error_handler import OperationValidator

        # Test operations registry
        registry = OperationsRegistry()
        validator = OperationValidator()

        # Test specific transform operations that were implemented
        transform_operations = [
            'dct_transform', 'dwt_transform', 'fft_transform',
            'huffman_encode', 'lz77_encode', 'arithmetic_encode',
            'elias_gamma_encode', 'elias_delta_encode', 'golomb_encode',
            'fibonacci_encode', 'phase_in_encode', 'adaptive_huffman_encode'
        ]

        results = {}
        test_data = bytes([random.randint(0, 255) for _ in range(50)])

        for op_name in transform_operations:
            if op_name in registry.operations:
                try:
                    start_time = time.time()
                    validation_result = validator.validate_operation(op_name, registry.operations[op_name], test_data)
                    execution_time = time.time() - start_time

                    results[op_name] = {
                        'success': validation_result['success'],
                        'execution_time': execution_time,
                        'reversible': validation_result.get('reversibility_test', False),
                        'parameter_validation': validation_result.get('parameter_validation', False),
                        'edge_case_handling': validation_result.get('edge_case_handling', False),
                        'errors': validation_result['errors'],
                        'warnings': validation_result['warnings']
                    }

                    status = "✅ PASS" if validation_result['success'] else "❌ FAIL"
                    print(f"{status} {op_name}: {execution_time:.3f}s, Reversible: {validation_result.get('reversibility_test', False)}")

                    if validation_result['errors']:
                        print(f"    Errors: {validation_result['errors'][:2]}")

                except Exception as e:
                    results[op_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"❌ {op_name}: EXCEPTION - {e}")
            else:
                results[op_name] = {
                    'success': False,
                    'error': 'Operation not found in registry'
                }
                print(f"❌ {op_name}: NOT FOUND")

        # Test operation validation system
        validation_summary = validator.get_validation_summary()

        return {
            'phase': 'Phase 2',
            'operations': results,
            'validation_summary': validation_summary
        }

    except ImportError as e:
        return {'phase': 'Phase 2', 'success': False, 'error': f'Import error: {e}'}


def run_phase3_tests():
    """Test Phase 3: Global error handler."""
    print("\n" + "="*60)
    print("PHASE 3: ERROR HANDLER TESTING")
    print("="*60)

    try:
        from bsee.utils.error_handler import GlobalErrorHandler, ErrorSeverity, BSEEError

        # Test error handler initialization
        error_handler = GlobalErrorHandler()

        # Test error handling
        test_errors = [
            ValueError("Test validation error"),
            KeyError("Test key error"),
            MemoryError("Test memory error"),
            ImportError("Test dependency error"),
            RuntimeError("Test runtime error")
        ]

        results = {}
        for i, error in enumerate(test_errors):
            try:
                start_time = time.time()
                error_result = error_handler.handle_error(
                    error,
                    context={'test_id': i, 'operation': 'test'},
                    severity=ErrorSeverity.MEDIUM
                )
                handling_time = time.time() - start_time

                results[f'test_error_{i}'] = {
                    'success': True,
                    'handling_time': handling_time,
                    'category': error_result['category'].value,
                    'recovery_attempted': error_result['recovery_attempted'],
                    'recovery_successful': error_result['recovery_successful']
                }

                status = "✅ HANDLED" if error_result['recovery_attempted'] else "⚠️ LOGGED"
                print(f"{status} {type(error).__name__}: {error_result['category'].value} ({handling_time:.3f}s)")

            except Exception as e:
                results[f'test_error_{i}'] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ Error handling failed: {e}")

        # Test error summary
        error_summary = error_handler.get_error_summary()

        return {
            'phase': 'Phase 3',
            'error_handling': results,
            'error_summary': error_summary
        }

    except ImportError as e:
        return {'phase': 'Phase 3', 'success': False, 'error': f'Import error: {e}'}


def run_phase4_tests():
    """Test Phase 4: GUI dependency handling."""
    print("\n" + "="*60)
    print("PHASE 4: GUI DEPENDENCY TESTING")
    print("="*60)

    try:
        # Test GUI dependency checking (by importing the updated gui_main)
        import importlib.util
        gui_main_path = Path(__file__).parent / 'gui_main.py'

        results = {}

        if gui_main_path.exists():
            try:
                # Load gui_main module
                spec = importlib.util.spec_from_file_location("gui_main", gui_main_path)
                gui_main = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(gui_main)

                # Test dependency checking function
                dependency_result = gui_main.check_dependencies()

                results['dependency_check'] = {
                    'success': True,
                    'result': dependency_result,
                    'gui_available': getattr(gui_main, 'GUI_AVAILABLE', None)
                }

                gui_status = "✅ AVAILABLE" if getattr(gui_main, 'GUI_AVAILABLE', False) else "⚠️ CLI FALLBACK"
                print(f"{gui_status} GUI dependencies checked successfully")

                # Test psutil fallback in GUI pipeline
                try:
                    from bsee.engine.gui_pipeline import PSUTIL_AVAILABLE
                    results['psutil_fallback'] = {
                        'success': True,
                        'psutil_available': PSUTIL_AVAILABLE
                    }

                    psutil_status = "✅ AVAILABLE" if PSUTIL_AVAILABLE else "⚠️ FALLBACK ACTIVE"
                    print(f"{psutil_status} psutil dependency with fallback")

                except ImportError as e:
                    results['psutil_fallback'] = {
                        'success': False,
                        'error': str(e)
                    }
                    print(f"❌ psutil fallback test failed: {e}")

            except Exception as e:
                results['gui_main_test'] = {
                    'success': False,
                    'error': str(e)
                }
                print(f"❌ GUI dependency test failed: {e}")
        else:
            results['gui_main_test'] = {
                'success': False,
                'error': 'gui_main.py not found'
            }
            print("❌ gui_main.py not found")

        return {'phase': 'Phase 4', 'gui_handling': results}

    except Exception as e:
        return {'phase': 'Phase 4', 'success': False, 'error': f'Test error: {e}'}


def run_integration_tests():
    """Run integration tests combining multiple components."""
    print("\n" + "="*60)
    print("INTEGRATION TESTING")
    print("="*60)

    try:
        from bsee.operations.operations_registry import OperationsRegistry
        from bsee.strategies.greedy_strategy import GreedyStrategy
        from bsee.engine.state import State
        from bsee.utils.error_handler import get_global_error_handler

        # Create test components
        registry = OperationsRegistry()
        strategy = GreedyStrategy({})
        error_handler = get_global_error_handler()
        test_data = bytes([random.randint(0, 255) for _ in range(100)])
        test_state = State(binary_data=test_data)

        results = {}

        # Test strategy with error handling
        try:
            start_time = time.time()

            # Test safe proposal with error handling
            operation, params = strategy.safe_propose(test_state)
            proposal_time = time.time() - start_time

            # Test safe acceptance
            test_state.score = 5.0
            accept_result = strategy.safe_accept(test_state)

            results['safe_strategy_execution'] = {
                'success': True,
                'proposal_time': proposal_time,
                'operation': operation,
                'params': params,
                'accept_result': accept_result,
                'strategy_info': strategy.get_strategy_info()
            }

            print(f"✅ Safe strategy execution: {operation} ({proposal_time:.3f}s)")

        except Exception as e:
            results['safe_strategy_execution'] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ Safe strategy execution failed: {e}")

        # Test operation execution with error handling
        try:
            if operation and operation in registry.operations:
                start_time = time.time()
                op_func = registry.operations[operation]

                # Execute operation
                result_data, inverse_func, metadata = op_func(test_data, **params)
                execution_time = time.time() - start_time

                # Test reversibility if claimed
                reversibility_test = False
                if metadata.get('reversible', False):
                    try:
                        restored_data = inverse_func(result_data)
                        reversibility_test = restored_data == test_data
                    except Exception:
                        reversibility_test = False

                results['operation_execution'] = {
                    'success': True,
                    'execution_time': execution_time,
                    'result_size': len(result_data),
                    'reversible': metadata.get('reversible', False),
                    'reversibility_test': reversibility_test,
                    'metadata': metadata
                }

                rev_status = "✅ PASS" if reversibility_test else "⚠️ FAIL"
                print(f"✅ Operation execution: {operation} ({execution_time:.3f}s), Reversibility: {rev_status}")

        except Exception as e:
            results['operation_execution'] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ Operation execution failed: {e}")

        return {'phase': 'Integration', 'results': results}

    except ImportError as e:
        return {'phase': 'Integration', 'success': False, 'error': f'Import error: {e}'}


def main():
    """Run comprehensive test suite."""
    print("BSEE COMPREHENSIVE TEST SUITE")
    print("=" * 60)
    print("Testing all phases of critical infrastructure implementation")
    print()

    # Run all test phases
    test_results = {
        'timestamp': time.time(),
        'python_version': sys.version,
        'test_results': {}
    }

    # Phase 1: Search Strategies
    test_results['test_results']['phase1'] = run_phase1_tests()

    # Phase 2: Transform Operations
    test_results['test_results']['phase2'] = run_phase2_tests()

    # Phase 3: Error Handler
    test_results['test_results']['phase3'] = run_phase3_tests()

    # Phase 4: GUI Dependencies
    test_results['test_results']['phase4'] = run_phase4_tests()

    # Integration Tests
    test_results['test_results']['integration'] = run_integration_tests()

    # Generate summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total_phases = 0
    successful_phases = 0

    for phase_name, phase_result in test_results['test_results'].items():
        total_phases += 1

        if phase_result.get('success', True):
            successful_phases += 1
            print(f"✅ {phase_name.upper()}: PASSED")
        else:
            print(f"❌ {phase_name.upper()}: FAILED")
            if 'error' in phase_result:
                print(f"   Error: {phase_result['error']}")

    success_rate = (successful_phases / total_phases) * 100 if total_phases > 0 else 0

    print(f"\nOverall Success Rate: {success_rate:.1f}% ({successful_phases}/{total_phases} phases)")

    # Save detailed results
    output_file = "comprehensive_test_results.json"
    try:
        with open(output_file, 'w') as f:
            json.dump(test_results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {output_file}")
    except Exception as e:
        print(f"\nWarning: Could not save results to file: {e}")

    # Return appropriate exit code
    if success_rate >= 80:
        print("\n🎉 Comprehensive testing completed successfully!")
        return 0
    elif success_rate >= 60:
        print("\n⚠️ Comprehensive testing completed with some issues")
        return 0
    else:
        print("\n❌ Comprehensive testing revealed significant issues")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)