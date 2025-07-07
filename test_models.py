import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv', parse_dates=['timestamp'])

# Import the models from enhanced_model
from enhanced_model import model1_linear, model2_demand_based, model3_competitive

def test_models():
    """Test all three pricing models"""
    print("Testing Dynamic Pricing Models")
    print("=" * 50)
    
    # Sample data for testing
    sample_data = df.head(10).copy()
    
    print("Sample data:")
    print(sample_data[['lot_id', 'occupancy', 'capacity', 'queue', 'traffic', 'is_special', 'vehicle_type']])
    print()
    
    # Test Model 1: Linear pricing
    print("Model 1: Linear Pricing Results")
    print("-" * 30)
    base_price = 10.0
    
    for idx, row in sample_data.iterrows():
        price1 = model1_linear(row['lot_id'], base_price, row['occupancy'], row['capacity'])
        print(f"Lot {row['lot_id']}: ${price1:.2f} (Occupancy: {row['occupancy']}/{row['capacity']})")
    
    print()
    
    # Test Model 2: Demand-based pricing
    print("Model 2: Demand-based Pricing Results")
    print("-" * 35)
    
    for idx, row in sample_data.iterrows():
        price2 = model2_demand_based(row)
        occupancy_rate = row['occupancy'] / row['capacity'] if row['capacity'] > 0 else 0
        print(f"Lot {row['lot_id']}: ${price2:.2f} (Occ: {occupancy_rate:.2f}, Queue: {row['queue']}, Traffic: {row['traffic']:.2f})")
    
    print()
    
    # Test Model 3: Competitive pricing
    print("Model 3: Competitive Pricing Results")
    print("-" * 35)
    
    # Get a batch of data from the same timestamp
    first_timestamp = sample_data['timestamp'].iloc[0]
    batch = df[df['timestamp'] == first_timestamp].copy()
    
    for idx, row in batch.iterrows():
        price3, message = model3_competitive(row, batch)
        print(f"Lot {row['lot_id']}: ${price3:.2f} - {message}")
    
    print()
    
    # Comparative analysis
    print("Comparative Analysis")
    print("-" * 20)
    
    comparison_data = []
    
    for idx, row in sample_data.head(5).iterrows():
        price1 = model1_linear(row['lot_id'], base_price, row['occupancy'], row['capacity'])
        price2 = model2_demand_based(row)
        price3, _ = model3_competitive(row, sample_data)
        
        comparison_data.append({
            'lot_id': row['lot_id'],
            'occupancy_rate': row['occupancy'] / row['capacity'] if row['capacity'] > 0 else 0,
            'queue': row['queue'],
            'model1_price': price1,
            'model2_price': price2,
            'model3_price': price3
        })
    
    comp_df = pd.DataFrame(comparison_data)
    print(comp_df.round(2))
    
    return comp_df

def analyze_pricing_trends():
    """Analyze pricing trends across different conditions"""
    print("\n" + "=" * 60)
    print("PRICING TREND ANALYSIS")
    print("=" * 60)
    
    # Sample different conditions
    analysis_data = []
    
    # High occupancy scenarios
    high_occ = df[df['occupancy'] / df['capacity'] > 0.8].head(10)
    
    # Low occupancy scenarios  
    low_occ = df[df['occupancy'] / df['capacity'] < 0.3].head(10)
    
    # High traffic scenarios
    high_traffic = df[df['traffic'] > 0.7].head(10)
    
    # Special day scenarios
    special_days = df[df['is_special'] == True].head(10)
    
    scenarios = {
        'High Occupancy': high_occ,
        'Low Occupancy': low_occ,
        'High Traffic': high_traffic,
        'Special Days': special_days
    }
    
    for scenario_name, scenario_data in scenarios.items():
        print(f"\n{scenario_name} Scenarios:")
        print("-" * (len(scenario_name) + 11))
        
        prices = []
        for idx, row in scenario_data.iterrows():
            price = model2_demand_based(row)
            prices.append(price)
            occupancy_rate = row['occupancy'] / row['capacity'] if row['capacity'] > 0 else 0
            print(f"  Lot {row['lot_id']}: ${price:.2f} (Occ: {occupancy_rate:.2f}, Queue: {row['queue']}, Traffic: {row['traffic']:.2f})")
        
        if prices:
            print(f"  Average Price: ${np.mean(prices):.2f}")
            print(f"  Price Range: ${min(prices):.2f} - ${max(prices):.2f}")

def validate_price_bounds():
    """Validate that all prices are within reasonable bounds"""
    print("\n" + "=" * 60)
    print("PRICE VALIDATION")
    print("=" * 60)
    
    all_prices = []
    violations = []
    
    # Test with random sample
    sample = df.sample(100)
    
    for idx, row in sample.iterrows():
        price = model2_demand_based(row)
        all_prices.append(price)
        
        # Check bounds (should be between $5 and $20 based on our model)
        if price < 5.0 or price > 20.0:
            violations.append({
                'lot_id': row['lot_id'],
                'price': price,
                'occupancy_rate': row['occupancy'] / row['capacity'] if row['capacity'] > 0 else 0,
                'queue': row['queue'],
                'traffic': row['traffic']
            })
    
    print(f"Tested {len(all_prices)} price calculations")
    print(f"Price range: ${min(all_prices):.2f} - ${max(all_prices):.2f}")
    print(f"Average price: ${np.mean(all_prices):.2f}")
    print(f"Standard deviation: ${np.std(all_prices):.2f}")
    print(f"Violations (outside $5-$20): {len(violations)}")
    
    if violations:
        print("\nViolations:")
        for v in violations:
            print(f"  Lot {v['lot_id']}: ${v['price']:.2f} (Occ: {v['occupancy_rate']:.2f})")

if __name__ == "__main__":
    # Run tests
    comp_df = test_models()
    analyze_pricing_trends()
    validate_price_bounds()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)