# Binary Structure Exploration Engine (BSEE)

BSEE is a comprehensive CLI-only Python program that analyzes binary files by applying reversible transformations to optimize user-specified metrics. The program operates entirely in the terminal with no GUI components, saving all results to timestamped text files for external analysis.

## 🚀 Major Infrastructure Improvements (Latest Update)

### Critical Infrastructure Fixes Implemented

**Search Strategy Overhaul:**
- **✅ MCTS Strategy**: Now implements real Monte Carlo Tree Search with complete tree data structure, UCT selection, expansion, simulation, and backpropagation phases
- **✅ Genetic Strategy**: Features real evolutionary algorithm with population management, crossover/mutation operators, and multiple selection methods
- **✅ Beam Search Strategy**: Implements true k-best candidate maintenance with dynamic width adjustment and diversity preservation
- **✅ Simulated Annealing Strategy**: Enhanced with temperature-aware operation selection and proper Boltzmann acceptance criteria
- **✅ Heuristic Strategy**: Multi-criteria evaluation system with cost, improvement, diversity, pattern, and history heuristics

**Configuration System:**
- **✅ YAML Integration**: All strategies now read their configuration from YAML files with 70+ parameters for MCTS alone
- **✅ Auto-config Creation**: Missing configuration files are automatically generated with sensible defaults
- **✅ Runtime Validation**: Parameter validation and fallback handling for all configurations

**Enhanced Features:**
- **✅ Error Handling**: Comprehensive error handling with graceful fallbacks and recovery strategies
- **✅ Memory Management**: Intelligent memory usage with tree pruning and resource monitoring
- **✅ Performance Optimization**: Efficient data structures and algorithms for large-scale analysis

## Features

- **100+ reversible binary operations** across 6 categories (including real mathematical transforms)
- **114 different metrics** across 9 measurement categories
- **6 search strategies** with REAL algorithm implementations (no more placeholder random.choice())
- **Dynamic cost modeling** with adaptive pricing
- **File Ideality metric** for measuring structural predictability
- **Complete operation history** for perfect reversibility
- **Extensive configuration** via YAML files with intelligent defaults
- **CLI interface** with comprehensive options
- **Robust error handling** and graceful degradation

## Installation

```bash
# Clone the repository
git clone https://github.com/bsee/bsee.git
cd bsee

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install BSEE in development mode
pip install -e .
```

## Startup Scripts

BSEE includes automated startup scripts that handle dependency installation and environment setup:

### Windows Scripts

- **`start.bat`** - Full setup with detailed checking and dependency installation
- **`start_simple.bat`** - Quick start with minimal checks

### Linux/macOS Scripts

- **`start.sh`** - Full setup for Unix-like systems

### What the Scripts Do

1. **Check Python Installation** - Verifies Python 3.9+ is available
2. **Create Virtual Environment** - Sets up isolated Python environment
3. **Install Dependencies** - Automatically installs required packages:
   - numpy, scipy, pyyaml (required)
   - lz4, zstandard (optional, for compression metrics)
4. **Activate Environment** - Ensures correct Python environment
5. **Run BSEE** - Starts the analysis with user-provided arguments

### Usage Examples

```bash
# Windows - interactive mode (shows help)
start.bat

# Windows - with arguments
start.bat "test.bin --strategy greedy --max-operations 50"

# Linux/macOS - interactive mode
./start.sh

# Linux/macOS - with arguments
./start.sh "test.bin --policy policy_ideality.yaml --strategy annealing"
```

## Quick Start

### Option 1: Using Startup Scripts (Recommended)

**Windows:**
```bash
# Automatic dependency installation and setup
start.bat

# Or with arguments
start.bat "test.bin --strategy greedy --max-operations 50"

# Quick start (minimal checks)
start_simple.bat
```

**Linux/macOS:**
```bash
# Make script executable and run
chmod +x start.sh
./start.sh

# Or with arguments
./start.sh "test.bin --strategy greedy --max-operations 50"
```

### Option 2: Manual Setup

```bash
# Clone the repository
git clone https://github.com/bsee/bsee.git
cd bsee

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install BSEE in development mode
pip install -e .
```

### Option 3: Run Directly

```bash
# Basic usage with File Ideality focus
python main.py input_file.bin \
  --policy config/policies/policy_ideality.yaml \
  --strategy greedy \
  --metrics file_ideality_score,entropy_global \
  --max-operations 1000

# Advanced usage with custom configuration
python main.py data/sample.bin \
  --policy config/policies/policy_custom.yaml \
  --costs config/costs/cost_custom.yaml \
  --strategy annealing \
  --metrics all \
  --target-metrics file_ideality_score=max,lz77_ratio=max \
  --max-operations 2000 \
  --max-cost 15000 \
  --allowed-ops xor,rotate,bitplane \
  --operation-limit 6 \
  --output-dir results/
```

## Command Line Options

- `input_file` - Binary file to analyze
- `--policy` - Policy YAML file defining metric weights
- `--costs` - Cost YAML file defining operation costs
- `--strategy` - Search strategy (greedy, beam, annealing, mcts, genetic, heuristic)
- `--metrics` - Comma-separated list of metrics or "all"
- `--target-metrics` - Which metrics to maximize/minimize
- `--max-operations` - Stop after N operations
- `--max-cost` - Stop when total cost exceeds this
- `--allowed-ops` - Only use these operations (optional constraint)
- `--operation-limit` - Maximum number of different operation types
- `--output-dir` - Where to save results

## Project Structure

```
bsee/
├── main.py                           # CLI entry point
├── config/                           # Configuration files
│   ├── policies/                     # Metric weight policies
│   ├── costs/                        # Operation cost definitions
│   └── strategies/                   # Search strategy configurations
├── bsee/                             # Main package
│   ├── engine/                       # Core execution engine
│   ├── operations/                   # Binary operations (100+ ops)
│   ├── metrics/                      # Analysis metrics (114 metrics)
│   ├── strategies/                   # Search strategies
│   ├── cost/                         # Cost modeling
│   ├── scoring/                      # State scoring
│   ├── results/                      # Results export
│   └── utils/                        # Utilities
├── tests/                            # Test suite
└── examples/                         # Example files and configs
```

## Operation Categories

1. **Bitwise Operations** - XOR, NOT, rotations (20 ops)
2. **Reordering Operations** - Permutations, shuffles (15 ops)
3. **Delta Operations** - Delta encoding variants (10 ops)
4. **Substitution Operations** - S-boxes, byte swaps (20 ops)
5. **Transform Operations** - BWT, bit planes (20 ops)
6. **Custom Operations** - User-defined operations (15+ ops)

## Metric Categories

1. **Entropy Metrics** - Shannon, conditional entropy (12 metrics)
2. **Compression Metrics** - LZ77, LZMA ratios (15 metrics)
3. **Pattern Metrics** - Autocorrelation, periodicity (10 metrics)
4. **Run Length Metrics** - Run statistics (8 metrics)
5. **Statistical Metrics** - Chi-square, KL divergence (12 metrics)
6. **Bitwise Metrics** - Bit plane analysis (15 metrics)
7. **Structure Metrics** - Alignment, block detection (10 metrics)
8. **Complexity Metrics** - Kolmogorov, LZ complexity (7 metrics)
9. **File Ideality Metrics** - Structural predictability (5 metrics)

## File Ideality

File Ideality measures how structured a binary is by testing if each bit is predictable from its surrounding context. Higher ideality indicates more structured, compressible, meaningful patterns, while lower ideality suggests random, noisy, encrypted, or compressed data.

## Output Files

All results are saved to timestamped directories:

```
results/run_20250106_143022/
├── summary.txt                      # Executive summary
├── final_binary.bin                 # Transformed binary
├── inverse_operations.json          # Complete reversal chain
├── full_timeline.csv                # Detailed operation log
├── metrics_comparison.txt           # Before/after metrics
├── operation_usage.txt              # Operation statistics
└── file_ideality_breakdown.txt      # Detailed ideality analysis
```

## Configuration

BSEE uses YAML files for configuration:

- **Policy Files** - Define metric weights and optimization targets
- **Cost Files** - Specify operation costs and modifiers
- **Strategy Files** - Configure search algorithm parameters

## Example Policies

```yaml
# config/policies/policy_ideality.yaml
name: "File Ideality Focus"
metric_weights:
  file_ideality_score: 100.0
  entropy_global: -5.0
  lz77_ratio: 10.0
targets:
  file_ideality_score: maximize
  entropy_global: minimize
```

## Testing

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/test_operations/
pytest tests/test_metrics/
pytest tests/test_integration/

# Run with coverage
pytest --cov=bsee
```

## License

MIT License - see LICENSE file for details.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## Support

For questions, issues, or contributions, please visit the GitHub repository.