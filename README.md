# Logistic Regression Analysis script

**Version 1.0**
### Creator: Juhani Merilehto - @juhanimerilehto - Jyväskylä University of Applied Sciences (JAMK), Likes institute

![JAMK Likes Logo](./assets/likes_str_logo.png)

## Overview

Logistic Regression script. This Python-based tool enables automated binary classification analysis using logistic regression. Developed for the Strategic Exercise Information and Research unit in Likes Institute, at JAMK University of Applied Sciences, this script provides comprehensive model evaluation including ROC curves, confusion matrices, feature importance analysis, and detailed classification reports.

## Features

- **Complete Model Pipeline**: Data preprocessing, model training, and evaluation
- **Comprehensive Metrics**: Accuracy, precision, recall, F1-score, ROC-AUC
- **Advanced Visualizations**: ROC curves, confusion matrices, feature importance plots
- **Model Diagnostics**: Probability distributions and prediction analysis
- **Excel Integration**: Detailed classification reports and model coefficients
- **Terminal Feedback**: Clear model performance metrics and statistics
- **Tested**: Tested with simulated data

## Hardware Requirements

- **Python:** 3.8 or higher
- **RAM:** 8GB recommended
- **Storage:** 1GB free space for analysis outputs
- **OS:** Windows 10/11, MacOS, or Linux
- **CPU:** Multi-core recommended for faster model training

## Installation

### 1. Clone the repository:
```bash
git clone https://github.com/juhanimerilehto/logistic-regression-analysis-script.git
cd logistic-regression-analysis-script
```

### 2. Create a virtual environment:
```bash
python -m venv stats-env
source stats-env/bin/activate  # For Windows: stats-env\Scripts\activate
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Basic usage:
```bash
python logisticregression.py
```

With custom parameters:
```bash
python logisticregression.py --excel_path "your_data.xlsx" --target "Target" --features "Feature1,Feature2,Feature3"
```

## Configuration Parameters

- `excel_path`: Path to Excel file (default: 'data.xlsx')
- `feature_columns`: List of predictor variables (default: ['Feature1', 'Feature2', 'Feature3'])
- `target_column`: Binary outcome variable (default: 'Target')
- `test_size`: Proportion of test data (default: 0.2)
- `random_state`: Random seed for reproducibility (default: 42)
- `output_prefix`: Prefix for output files (default: 'logistic')

## File Structure

```plaintext
├── logistic-regression-analysis-script/
│   ├── assets/
│   │   └── likes_str_logo.png
│   ├── logisticregression.py
│   ├── requirements.txt
│   └── README.md
```

## Credits

- **Juhani Merilehto (@juhanimerilehto)** – Specialist, Data and Statistics
- **JAMK Likes** – Organization sponsor

## License

This project is licensed for free use under the condition that proper credit is given to Juhani Merilehto (@juhanimerilehto) and JAMK Likes institute. You are free to use, modify, and distribute this project, provided that you mention the original author and institution and do not hold them liable for any consequences arising from the use of the software.