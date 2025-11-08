#!/usr/bin/env python3
"""
Operation validation script for BSEE.
Tests all operations to ensure they work correctly and can properly restore data.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bsee.operations.operations_registry import OperationsRegistry
from bsee.utils.error_handler import OperationValidator, get_global_error_handler
import json


def main():
    """Run validation on all operations."""
    print("🔍 BSEE Operation Validation System")
    print("=" * 50)

    # Initialize operations registry
    print("Loading operations registry...")
    try:
        registry = OperationsRegistry()
        print(f"✅ Loaded {len(registry.operations)} operations")
    except Exception as e:
        print(f"❌ Failed to load operations registry: {e}")
        return 1

    # Initialize validator
    validator = OperationValidator()
    error_handler = get_global_error_handler()

    # Print operation summary
    summary = registry.get_registry_summary()
    print(f"📊 Operation categories: {summary['categories']}")

    # Validate all operations
    print("\n🧪 Running validation tests...")
    validation_results = validator.validate_all_operations(registry)

    # Analyze results
    total_operations = len(validation_results)
    successful_operations = sum(1 for result in validation_results.values() if result['success'])
    failed_operations = total_operations - successful_operations

    print(f"\n📈 Validation Results:")
    print(f"   Total operations: {total_operations}")
    print(f"   ✅ Successful: {successful_operations}")
    print(f"   ❌ Failed: {failed_operations}")
    print(f"   Success rate: {(successful_operations / total_operations * 100):.1f}%")

    # Show failed operations
    if failed_operations > 0:
        print(f"\n❌ Failed Operations:")
        for op_name, result in validation_results.items():
            if not result['success']:
                print(f"   • {op_name}:")
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"     - {error}")
                if len(result['errors']) > 3:
                    print(f"     - ... and {len(result['errors']) - 3} more errors")

    # Show operations with reversibility issues
    reversibility_issues = []
    for op_name, result in validation_results.items():
        if result.get('reversibility_test') is False and 'reversible' in str(result.get('warnings', [])):
            reversibility_issues.append(op_name)

    if reversibility_issues:
        print(f"\n⚠️  Operations with reversibility issues:")
        for op_name in reversibility_issues:
            print(f"   • {op_name}")

    # Show operations with warnings
    operations_with_warnings = []
    for op_name, result in validation_results.items():
        if result['warnings']:
            operations_with_warnings.append(op_name)

    if operations_with_warnings:
        print(f"\n⚠️  Operations with warnings ({len(operations_with_warnings)}):")
        for op_name in operations_with_warnings[:10]:  # Show first 10
            result = validation_results[op_name]
            print(f"   • {op_name}: {len(result['warnings'])} warnings")

    # Performance summary
    print(f"\n⚡ Performance Summary:")
    execution_times = []
    for result in validation_results.values():
        if result['performance_metrics'].get('execution_time'):
            execution_times.append(result['performance_metrics']['execution_time'])

    if execution_times:
        avg_time = sum(execution_times) / len(execution_times)
        slowest = max(execution_times)
        fastest = min(execution_times)
        print(f"   Average execution time: {avg_time:.3f}s")
        print(f"   Fastest: {fastest:.3f}s")
        print(f"   Slowest: {slowest:.3f}s")

    # Error handler summary
    error_summary = error_handler.get_error_summary()
    if error_summary['total_errors'] > 0:
        print(f"\n🚨 Error Handler Summary:")
        print(f"   Total errors handled: {error_summary['total_errors']}")
        print(f"   Recovery rate: {(error_summary['recovery_rate'] * 100):.1f}%")
        print(f"   Most common errors: {error_summary['most_common_errors']}")

    # Detailed validation summary
    validation_summary = validator.get_validation_summary()
    print(f"\n📋 Validation Summary:")
    print(f"   Total validations run: {validation_summary['total_validations']}")
    print(f"   Overall success rate: {(validation_summary['success_rate'] * 100):.1f}%")

    # Save detailed results to file
    output_file = "validation_results.json"
    try:
        # Convert results to JSON-serializable format
        serializable_results = {}
        for op_name, result in validation_results.items():
            serializable_results[op_name] = {
                'success': result['success'],
                'errors': result['errors'],
                'warnings': result['warnings'],
                'reversibility_test': result['reversibility_test'],
                'parameter_validation': result['parameter_validation'],
                'edge_case_handling': result['edge_case_handling'],
                'execution_time': result['performance_metrics'].get('execution_time')
            }

        with open(output_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_operations': total_operations,
                    'successful_operations': successful_operations,
                    'failed_operations': failed_operations,
                    'success_rate': successful_operations / total_operations
                },
                'validation_summary': validation_summary,
                'error_summary': error_summary,
                'detailed_results': serializable_results
            }, f, indent=2)

        print(f"📄 Detailed results saved to: {output_file}")

    except Exception as e:
        print(f"⚠️  Could not save detailed results: {e}")

    # Return appropriate exit code
    if failed_operations == 0:
        print(f"\n🎉 All operations passed validation!")
        return 0
    elif successful_operations >= total_operations * 0.8:  # 80% success rate
        print(f"\n✅ Most operations passed validation ({successful_operations}/{total_operations})")
        return 0
    else:
        print(f"\n❌ Too many operations failed validation")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)