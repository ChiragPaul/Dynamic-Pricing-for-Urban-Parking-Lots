import pandas as pd
import numpy as np

# Load the dataset
df = pd.read_csv('dataset.csv', parse_dates=['timestamp'])

# Model parameters
α = 5.0  # occupancy sensitivity  
β = 2.0  # queue sensitivity
γ = 1.0  # traffic sensitivity
δ = 3.0  # special day multiplier
λ = 0.5  # demand adjustment factor
veh_weight = {'car': 1.0, 'bike': -0.5, 'truck': 1.5}

def proximity(lat1, lon1, lat2, lon2):
    """Calculate simple distance between two points"""
    return np.sqrt((lat1-lat2)**2 + (lon1-lon2)**2)

def model1_linear(lot_id, prev_price, occupancy, capacity):
    """Model 1: Simple linear pricing based on occupancy"""
    occupancy_rate = occupancy / capacity if capacity > 0 else 0
    return prev_price + α * occupancy_rate

def model2_demand_based(row):
    """Model 2: Demand-based pricing with multiple factors"""
    base = 10.0
    
    occ = row['occupancy']
    cap = row['capacity']
    queue = row['queue']
    traffic = row['traffic']
    special = row['is_special']
    vtype = row['vehicle_type']
    
    # Calculate demand score
    occupancy_rate = occ / cap if cap > 0 else 0
    demand = (α * occupancy_rate + 
              β * queue - 
              γ * traffic + 
              δ * special + 
              veh_weight.get(vtype, 0))
    
    # Normalize demand using tanh to prevent extreme values
    norm = np.tanh(demand / 10)
    price = base * (1 + λ * norm)
    
    # Ensure price bounds
    price = np.clip(price, base * 0.5, base * 2)
    
    return round(price, 2)

def model3_competitive(row, lot_data_batch):
    """Model 3: Competitive pricing with rerouting logic"""
    base_price = model2_demand_based(row)
    
    # Get nearby lots (within 0.01 degrees ~1km)
    current_lat, current_lon = row['lat'], row['lon']
    lot_id = row['lot_id']
    
    nearby_lots = []
    for idx, other_lot in lot_data_batch.iterrows():
        if other_lot['lot_id'] != lot_id:
            dist = proximity(current_lat, current_lon, 
                           other_lot['lat'], other_lot['lon'])
            if dist < 0.01:  # Within ~1km
                other_price = model2_demand_based(other_lot)
                nearby_lots.append({
                    'lot_id': other_lot['lot_id'],
                    'distance': dist,
                    'price': other_price,
                    'occupancy_rate': other_lot['occupancy'] / other_lot['capacity']
                })
    
    if not nearby_lots:
        return base_price, 'no nearby competitors'
    
    # Competitive logic
    occupancy_rate = row['occupancy'] / row['capacity'] if row['capacity'] > 0 else 0
    cheaper_lots = [lot for lot in nearby_lots if lot['price'] < base_price]
    
    if occupancy_rate > 0.8 and cheaper_lots:
        # High occupancy and cheaper alternatives exist - suggest reroute
        adjusted_price = base_price * 0.9
        return adjusted_price, f'reroute suggested - {len(cheaper_lots)} cheaper alternatives'
    elif not cheaper_lots and occupancy_rate < 0.9:
        # No cheaper alternatives, can increase price
        adjusted_price = min(base_price * 1.05, base_price * 1.2)
        return adjusted_price, 'premium pricing - no cheaper alternatives'
    else:
        return base_price, 'standard pricing'

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

def simulate_real_time_pricing():
    """Simulate real-time pricing for a specific day"""
    print("\n" + "=" * 60)
    print("REAL-TIME PRICING SIMULATION")
    print("=" * 60)
    
    # Get data for first day
    first_day = df['timestamp'].dt.date.iloc[0]
    day_data = df[df['timestamp'].dt.date == first_day].copy()
    
    print(f"Simulating pricing for {first_day}")
    print(f"Processing {len(day_data)} records across {day_data['lot_id'].nunique()} lots")
    print()
    
    # Group by timestamp to simulate real-time batches
    for timestamp, batch in day_data.groupby('timestamp'):
        print(f"Time: {timestamp.strftime('%H:%M:%S')}")
        print("-" * 20)
        
        # Calculate competitive pricing for this batch
        competitive_results = []
        
        for idx, row in batch.iterrows():
            price2 = model2_demand_based(row)
            price3, message = model3_competitive(row, batch)
            
            competitive_results.append({
                'lot_id': row['lot_id'],
                'occupancy_rate': row['occupancy'] / row['capacity'] if row['capacity'] > 0 else 0,
                'demand_price': price2,
                'competitive_price': price3,
                'message': message
            })
        
        # Sort by competitive price
        competitive_results.sort(key=lambda x: x['competitive_price'])
        
        # Display results
        for result in competitive_results:
            print(f"  Lot {result['lot_id']:2d}: ${result['competitive_price']:5.2f} "
                  f"(Occ: {result['occupancy_rate']:.2f}) - {result['message']}")
        
        print()
        
        # Break after first few timestamps for demo
        if len([t for t, _ in day_data.groupby('timestamp') if t <= timestamp]) >= 3:
            break

if __name__ == "__main__":
    # Run tests
    comp_df = test_models()
    analyze_pricing_trends()
    simulate_real_time_pricing()
    
    print("\n" + "=" * 60)
    print("TESTING COMPLETED")
    print("=" * 60)