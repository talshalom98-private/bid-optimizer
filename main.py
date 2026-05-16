"""
Entry point for the bid optimizer.
The full algorithm, explanations, and reasoning live in bid_optimizer.ipynb.
This script imports the core optimizer and runs it on campaign_data.csv.
Requires Python >= 3.9.
"""
import sys

if sys.version_info < (3, 9):
    sys.exit(
        f"Python 3.9+ required (found {sys.version}). "
        "Run: py -3.9 main.py  or  python3 main.py"
    )

import pandas as pd
from bid_optimizer import BidOptimizer


def main():
    data = pd.read_csv("campaign_data.csv")
    optimizer = BidOptimizer(data)
    recommendations = optimizer.optimize()
    print(recommendations.to_string(index=False))
    recommendations.to_csv("bid_recommendations.csv", index=False)
    print("\nSaved to bid_recommendations.csv")


if __name__ == "__main__":
    main()
