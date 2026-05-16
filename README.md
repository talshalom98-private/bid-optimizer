# Bid Optimizer — Algorithm Developer Home Assignment

## Overview
A keyword bid optimization system for e-commerce advertising campaigns, targeting ROAS (Return on Ad Spend) improvement while respecting budget and bid constraints.

## Project Structure
```
bid-optimizer/
├── bid_optimizer.ipynb   # Main notebook: full solution with explanations
├── main.py               # CLI entry point: runs the optimizer and prints results
├── campaign_data.csv     # Input: 30-day historical keyword performance data
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup
```bash
pip install -r requirements.txt
```

## Running the Optimizer
```bash
python main.py
```
Outputs recommended bids for each keyword to stdout and saves `bid_recommendations.csv`.

## Full Walkthrough
Open `bid_optimizer.ipynb` in Jupyter for the complete explanation of the algorithm, assumptions, and reasoning:
```bash
jupyter notebook bid_optimizer.ipynb
```

## Approach Summary
- **Part A**: Bid optimization algorithm using ROAS-based bid scaling with statistical confidence weighting for low-data keywords, subject to campaign budget constraints and bid bounds [$0.20, $15.00].
- **Part B**: Uncertainty handling — conservative bid adjustment for keywords with insufficient data.
- **Part C**: Extension — see notebook for the chosen extension and implementation.

## Assumptions & Trade-offs
See the notebook cells under each part for detailed reasoning.
