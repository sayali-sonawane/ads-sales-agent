# LinkedIn Ads Sales Agent

LangGraph-powered system that generates professional LinkedIn advertising sales pitches with data analysis, visualizations, and PowerPoint presentations.

## Features

✅ **8 Industries**: Technology, Healthcare, Finance, Retail, Manufacturing, Education, Real Estate, Automotive  
✅ **Data Generation**: Realistic customer profiles, performance trends, competitor analysis  
✅ **Professional Visualizations**: Performance trends, benchmark comparisons, competitive analysis  
✅ **AI Recommendations**: GPT-3.5-turbo powered strategic recommendations (with fallback logic)  
✅ **PowerPoint Output**: Complete sales pitch presentations  

## Quick Start

```bash
# Setup
source venv/bin/activate
pip install -r requirements.txt

# Optional: Add OpenAI API key
cp .env.example .env
# Edit .env with your OPENAI_API_KEY

# Generate sample data (first time)
python -m sales_agent.main --generate-data

# Run sales agent
python -m sales_agent.main
```

## Commands

- `python -m sales_agent.main` - Run complete workflow
- `python -m sales_agent.main --generate-data` - Generate sample datasets
- `python -m sales_agent.main --demo` - Multi-industry demo
- `python -m sales_agent.main --help` - Show help

## Generated Outputs

- **CSV Files**: Customer profiles, performance data, competitor analysis
- **Charts**: Performance trends, benchmark comparisons, competitor analysis  
- **PowerPoint**: Complete sales pitch presentation with recommendations

## Architecture

**LangGraph Workflow**: Data Preparation → Performance Analysis → Visualization → AI Recommendations → Presentation

**Key Components**:
- `models.py` - Data structures for customers, performance, competitors
- `data_generator.py` - Realistic sample data creation
- `agents.py` - Specialized agents for analysis, visualization, recommendations
- `workflow.py` - LangGraph orchestration and state management

## Development

```bash
# Tests
make test

# Code quality
make format
make lint

# Clean
make clean
```

## Requirements

- Python 3.9+
- OpenAI API key (optional - works with fallback recommendations)
- Dependencies in requirements.txt
