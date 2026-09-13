#!/usr/bin/env python3
"""Deterministic Synthetic Retail Time-Series Data Generator for GlassBox-BI.

Generates reproducible, realistic business time-series demand data for development
and testing without hard-coding to any specific vendor (e.g., Walmart).

Zero Data Leakage Rule: Features are never computed from future target values.
"""

import argparse
import math
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import pandas as pd


def generate_retail_dataset(
    num_rows: int = 50000,
    seed: int = 42,
    start_date_str: str = "2024-01-01",
) -> pd.DataFrame:
    """Generates a realistic, deterministic synthetic retail time-series dataset.

    Args:
        num_rows: Exact number of data rows to generate (default 50,000).
        seed: Fixed random seed for complete reproducibility.
        start_date_str: Initial observation date (YYYY-MM-DD).

    Returns:
        pd.DataFrame containing the generated business records.
    """
    rng = np.random.default_rng(seed)

    # 1. Define Entity, Product, and Hierarchy Dimensions
    num_entities = 10
    num_products = 20
    total_series = num_entities * num_products  # 200 unique (entity, product) pairs
    days_per_series = math.ceil(num_rows / total_series)  # 250 days

    entities = [f"STORE_{i:03d}" for i in range(1, num_entities + 1)]
    products = [f"PROD_{i:03d}" for i in range(1, num_products + 1)]

    categories = ["Beverages", "Snacks", "Pantry", "Personal Care"]
    regions = ["North", "South", "East", "West"]
    store_types = ["Supermarket", "Express", "Flagship"]

    # Assign persistent attributes to entities and products
    entity_metadata = {}
    for i, ent in enumerate(entities):
        entity_metadata[ent] = {
            "region": regions[i % len(regions)],
            "store_type": store_types[i % len(store_types)],
            "base_multiplier": 0.8 + 0.5 * (i / max(1, num_entities - 1)),
        }

    product_metadata = {}
    for j, prod in enumerate(products):
        cat = categories[j % len(categories)]
        base_price = round(2.50 + 2.25 * j, 2)
        product_metadata[prod] = {
            "category": cat,
            "base_price": base_price,
            "base_demand": 15.0 + 2.0 * (j % 5),
            "elasticity": 0.75 + 0.1 * (j % 3),
        }

    # 2. Generate Temporal Calendar
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
    dates = [start_date + timedelta(days=d) for d in range(days_per_series)]

    # Weekly seasonality factors (Mon=0 to Sun=6: weekend bump)
    dow_factors = [0.90, 0.92, 0.95, 0.98, 1.15, 1.35, 1.25]

    # Pre-identify common retail holiday periods (approx 5% frequency)
    holiday_days = set()
    for d_idx, dt in enumerate(dates):
        # Specific holidays: New Year (Jan 1), Easter-time, Memorial Day, July 4, Labor Day
        if (dt.month == 1 and dt.day in (1, 2)) or \
           (dt.month == 7 and dt.day in (3, 4)) or \
           (dt.month == 5 and 25 <= dt.day <= 31 and dt.weekday() == 0) or \
           (dt.month == 9 and 1 <= dt.day <= 7 and dt.weekday() == 0) or \
           (dt.month == 11 and 24 <= dt.day <= 30 and dt.weekday() == 3):
            holiday_days.add(dt)

    # 3. Build Deterministic Time-Series Grid
    records = []
    current_count = 0

    for d_idx, dt in enumerate(dates):
        dow = dt.weekday()
        doy = dt.timetuple().tm_yday
        is_holiday = 1 if dt in holiday_days else 0

        # Yearly / harmonic seasonality wave
        yearly_wave = 1.0 + 0.12 * math.sin(2 * math.pi * doy / 365.25) + 0.06 * math.cos(4 * math.pi * doy / 365.25)
        dow_mult = dow_factors[dow]

        for ent in entities:
            ent_meta = entity_metadata[ent]
            ent_mult = ent_meta["base_multiplier"]

            for prod in products:
                if current_count >= num_rows:
                    break

                prod_meta = product_metadata[prod]
                base_price = prod_meta["base_price"]
                base_demand = prod_meta["base_demand"]
                elasticity = prod_meta["elasticity"]

                # Promotion assignment: weekend campaigns or occasional weekday promotions (~15% rate)
                promo_prob = 0.25 if dow in (4, 5) else 0.10
                is_promo = 1 if rng.random() < promo_prob else 0

                # Price calculation with slight promotional discounting (-15% on promo)
                if is_promo:
                    price = round(base_price * (0.85 + 0.05 * rng.random()), 2)
                else:
                    price = round(base_price * (0.98 + 0.04 * rng.random()), 2)

                # Price elasticity effect
                price_ratio = base_price / max(0.1, price)
                price_effect = price_ratio ** elasticity

                # Promotional and holiday demand boosts
                promo_lift = 1.35 if is_promo else 1.0
                holiday_lift = 1.45 if is_holiday else 1.0

                # Upward business trend
                trend = 0.03 * d_idx

                # Deterministic noise with controlled variance
                noise = rng.normal(loc=1.0, scale=0.12)
                noise = max(0.65, min(1.45, noise))

                # Aggregate Multi-Component Demand Equation
                continuous_demand = (
                    base_demand
                    * ent_mult
                    * dow_mult
                    * yearly_wave
                    * price_effect
                    * promo_lift
                    * holiday_lift
                    * noise
                    + trend
                )
                target = int(max(0, round(continuous_demand)))

                # Inventory: modeled based on base capacity and historical buffer
                # Zero Leakage Rule: Inventory does NOT use future target values.
                # It is a function of typical demand capacity plus safety stock.
                expected_safety_stock = int(base_demand * ent_mult * 3.5)
                stock_noise = rng.integers(-5, 15)
                inventory = max(10, expected_safety_stock + stock_noise)

                records.append({
                    "date": dt.isoformat(),
                    "entity_id": ent,
                    "product_id": prod,
                    "category": prod_meta["category"],
                    "region": ent_meta["region"],
                    "store_type": ent_meta["store_type"],
                    "target": target,
                    "price": price,
                    "promotion": is_promo,
                    "holiday": is_holiday,
                    "inventory": inventory,
                })
                current_count += 1

            if current_count >= num_rows:
                break
        if current_count >= num_rows:
            break

    df = pd.DataFrame(records)
    assert len(df) == num_rows, f"Expected exactly {num_rows} rows, got {len(df)}"
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic retail time-series data for GlassBox-BI.")
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/synthetic/retail_50k.csv",
        help="Path for full synthetic CSV output.",
    )
    parser.add_argument(
        "--sample-output",
        type=str,
        default="data/sample/retail_sample.csv",
        help="Path for lightweight sample CSV output.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=50000,
        help="Number of rows for the full dataset (default: 50,000).",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=150,
        help="Number of rows for the sample dataset (default: 150).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Generate full dataset
    print(f"Generating deterministic synthetic dataset with {args.rows} rows (seed={args.seed})...")
    full_df = generate_retail_dataset(num_rows=args.rows, seed=args.seed)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(out_path, index=False)
    print(f"[SUCCESS] Wrote {len(full_df)} rows to: {out_path} ({out_path.stat().st_size / (1024*1024):.2f} MB)")

    # Generate lightweight sample dataset (exact same schema)
    sample_df = full_df.head(args.sample_rows).copy()
    sample_path = Path(args.sample_output)
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_df.to_csv(sample_path, index=False)
    print(f"[SUCCESS] Wrote sample {len(sample_df)} rows to: {sample_path} ({sample_path.stat().st_size / 1024:.2f} KB)")


if __name__ == "__main__":
    main()
