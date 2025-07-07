import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate sample dataset for 14 parking lots over 73 days
np.random.seed(42)

# Parameters
num_lots = 14
num_days = 73
times_per_day = 18  # 8:00 AM to 4:30 PM with 30 min intervals

# Generate lot IDs and their fixed properties
lots = []
for i in range(num_lots):
    lot_data = {
        'lot_id': i + 1,
        'lat': 40.7128 + np.random.normal(0, 0.01),  # Around NYC
        'lon': -74.0060 + np.random.normal(0, 0.01),
        'capacity': np.random.randint(20, 101)  # 20-100 spaces
    }
    lots.append(lot_data)

# Generate time series data
data = []
start_date = datetime(2024, 1, 1, 8, 0)  # Start at 8:00 AM

for day in range(num_days):
    for time_slot in range(times_per_day):
        timestamp = start_date + timedelta(days=day, minutes=time_slot*30)
        
        # Special days (weekends, holidays)
        is_special = timestamp.weekday() >= 5 or np.random.random() < 0.05
        
        # Traffic pattern (higher during rush hours)
        hour = timestamp.hour
        if 8 <= hour <= 9 or 17 <= hour <= 18:
            traffic_base = 0.8
        elif 12 <= hour <= 13:
            traffic_base = 0.6
        else:
            traffic_base = 0.3
        
        traffic = np.clip(traffic_base + np.random.normal(0, 0.2), 0, 1)
        
        for lot in lots:
            # Occupancy based on time of day and special events
            occupancy_rate = 0.3 + 0.4 * traffic + (0.2 if is_special else 0)
            occupancy_rate = np.clip(occupancy_rate + np.random.normal(0, 0.1), 0, 1)
            occupancy = int(occupancy_rate * lot['capacity'])
            
            # Queue length (higher when occupancy is high)
            queue_prob = max(0, (occupancy_rate - 0.7) * 3)
            queue = np.random.poisson(queue_prob)
            
            # Vehicle type
            vehicle_type = np.random.choice(['car', 'bike', 'truck'], p=[0.7, 0.2, 0.1])
            
            record = {
                'lot_id': lot['lot_id'],
                'lat': lot['lat'],
                'lon': lot['lon'],
                'capacity': lot['capacity'],
                'occupancy': occupancy,
                'queue': queue,
                'traffic': traffic,
                'is_special': is_special,
                'vehicle_type': vehicle_type,
                'timestamp': timestamp
            }
            data.append(record)

# Create DataFrame and save
df = pd.DataFrame(data)
df = df.sort_values(['timestamp', 'lot_id'])
df.to_csv('dataset.csv', index=False)
print(f"Dataset created with {len(df)} records")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"Number of lots: {df['lot_id'].nunique()}")