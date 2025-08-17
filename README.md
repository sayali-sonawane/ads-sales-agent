# Sales Agent

A Python-based sales agent application designed to help with sales automation and management.

## Features

- Modular Python package structure
- Comprehensive testing framework
- Code quality tools (linting, formatting, type checking)
- Easy development workflow with Makefile commands

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sayali-sonawane/ads-sales-agent.git
cd ads-sales-agent
```

### 2. Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
# venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install the package in development mode and all dependencies
make install
```

This command will:
- Install the `sales-agent` package in development mode
- Install all required dependencies from `requirements.txt`

### 4. Run the Application

```bash
# Run the sales agent application
make run
```

## Development Commands

The project includes a Makefile with convenient commands for development:

```bash
# Show all available commands
make help

# Run tests
make test

# Run tests with coverage
make test-cov

# Format code with black
make format

# Check code formatting
make format-check

# Run linting checks
make lint

# Clean build artifacts
make clean
```

## Project Structure

```
sales-agent/
├── setup.py                 # Package configuration
├── requirements.txt         # Project dependencies
├── pytest.ini             # Test configuration
├── .gitignore             # Git ignore patterns
├── Makefile               # Development commands
├── sales_agent/           # Main package
│   ├── __init__.py        # Package initialization
│   └── main.py            # Main application entry point
└── tests/                 # Test directory
    ├── __init__.py        # Test package initialization
    └── test_main.py       # Basic test file
```

## Dependencies

### Core Dependencies
- Add your project-specific dependencies in `setup.py`

### Development Dependencies
- **Testing**: pytest, pytest-cov
- **Code Quality**: black (formatter), flake8 (linting), mypy (type checking)
- **Development Tools**: pre-commit

## Running Tests

```bash
# Run all tests
make test

# Run tests with coverage report
make test-cov
```

## Code Quality

```bash
# Format code automatically
make format

# Check if code is properly formatted
make format-check

# Run linting and type checking
make lint
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and quality checks (`make test`, `make lint`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## License

This project is licensed under the MIT License.

## Support

For questions or issues, please open an issue on GitHub or contact the development team.

## What's Next?

This is a basic project structure. You can now:

1. **Add your sales agent logic** in `sales_agent/main.py`
2. **Create additional modules** in the `sales_agent/` package
3. **Add more tests** in the `tests/` directory
4. **Customize dependencies** in `requirements.txt`
5. **Configure additional tools** as needed

The project is set up with best practices and ready for you to build your sales agent application!
