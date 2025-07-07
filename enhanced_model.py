import pathway as pw
import pandas as pd
import numpy as np
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import ColumnDataSource
from bokeh.layouts import gridplot
import time
import os
from collections import defaultdict

# Load or create dataset
if not os.path.exists('dataset.csv'):
    print("Dataset not found. Creating sample dataset...")
    exec(open('create_dataset.py').read())

df = pd.read_csv('dataset.csv', parse_dates=['timestamp'])
cols = ['lot_id','lat','lon','capacity','occupancy','queue','traffic','is_special','vehicle_type','timestamp']
df = df[cols]
print(f"Loaded {len(df)} records from dataset")

# Configuration
lots = df.lot_id.unique()
base_price = {lot: 10.0 for lot in lots}
price_history = defaultdict(list)

# Model parameters
α = 5.0  # occupancy sensitivity  
β = 2.0  # queue sensitivity
γ = 1.0  # traffic sensitivity
δ = 3.0  # special day multiplier
λ = 0.5  # demand adjustment factor
veh_weight = {'car': 1.0, 'bike': -0.5, 'truck': 1.5}

def stream_batches(df, delay=0.5):
    """Stream data in batches by timestamp"""
    for timestamp, batch in df.groupby('timestamp'):
        yield batch
        time.sleep(delay)

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
    for _, other_lot in lot_data_batch.iterrows():
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

# Pathway streaming setup
class BatchSource(pw.io.python.ConnectorSubject):
    def __init__(self, iterator):
        super().__init__()
        self.it = iterator
    
    def run(self):
        for batch in self.it:
            self.next_df(batch)
            time.sleep(0.1)

class ParkingSchema(pw.Schema):
    lot_id: int
    lat: float
    lon: float
    capacity: int
    occupancy: int
    queue: int
    traffic: float
    is_special: bool
    vehicle_type: str

# Create streaming source
source = pw.io.python.read(BatchSource(stream_batches(df)), schema=ParkingSchema)

# Data transformation
t = source.select(
    lot_id=pw.this.lot_id,
    lat=pw.this.lat,
    lon=pw.this.lon,
    capacity=pw.this.capacity,
    occupancy=pw.this.occupancy,
    queue=pw.this.queue,
    traffic=pw.this.traffic,
    special=pw.this.is_special,
    vehicle=pw.this.vehicle_type
)

# Pricing UDF
@pw.udf
def calculate_price_udf(row):
    """Calculate price using Model 2 (demand-based)"""
    row_dict = {
        'occupancy': row['occupancy'],
        'capacity': row['capacity'],
        'queue': row['queue'],
        'traffic': row['traffic'],
        'is_special': row['special'],
        'vehicle_type': row['vehicle']
    }
    return model2_demand_based(row_dict)

# Create pricing stream
priced_stream = t.select(
    lot_id=pw.this.lot_id,
    lat=pw.this.lat,
    lon=pw.this.lon,
    capacity=pw.this.capacity,
    occupancy=pw.this.occupancy,
    queue=pw.this.queue,
    traffic=pw.this.traffic,
    special=pw.this.special,
    vehicle=pw.this.vehicle,
    timestamp=pw.now(),
    price=calculate_price_udf(pw.this)
)

# Visualization setup
plot_data = defaultdict(lambda: ColumnDataSource(data=dict(timestamp=[], price=[], occupancy_rate=[])))

def create_visualizations(df_batch):
    """Create real-time visualizations"""
    plots = []
    
    if df_batch.empty:
        return
    
    print(f"Processing batch with {len(df_batch)} records at {pd.Timestamp.now()}")
    
    for lot_id in df_batch['lot_id'].unique():
        df_lot = df_batch[df_batch['lot_id'] == lot_id]
        
        if df_lot.empty:
            continue
            
        source = plot_data[lot_id]
        
        # Calculate occupancy rate
        occupancy_rates = df_lot['occupancy'] / df_lot['capacity']
        
        # Prepare new data
        new_data = {
            'timestamp': pd.to_datetime(df_lot['timestamp']),
            'price': df_lot['price'],
            'occupancy_rate': occupancy_rates
        }
        
        # Stream new data (with rollover to keep plot manageable)
        try:
            source.stream(new_data, rollover=200)
        except Exception as e:
            print(f"Error streaming data for lot {lot_id}: {e}")
            continue
        
        # Create plot
        p = figure(
            title=f"Lot {lot_id} - Pricing vs Occupancy",
            x_axis_type='datetime',
            width=400,
            height=300,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )
        
        # Price line
        p.line('timestamp', 'price', source=source, line_width=2, 
               color='blue', legend_label='Price ($)')
        
        # Occupancy rate line (secondary y-axis effect)
        p.line('timestamp', 'occupancy_rate', source=source, line_width=2, 
               color='red', alpha=0.7, legend_label='Occupancy Rate')
        
        p.xaxis.axis_label = "Time"
        p.yaxis.axis_label = "Price ($) / Occupancy Rate"
        p.legend.location = "top_left"
        
        plots.append(p)
        
        # Print some statistics
        if len(df_lot) > 0:
            avg_price = df_lot['price'].mean()
            avg_occupancy = occupancy_rates.mean()
            print(f"Lot {lot_id}: Avg Price=${avg_price:.2f}, Avg Occupancy={avg_occupancy:.2f}")
    
    if plots:
        try:
            show(gridplot(plots, ncols=2))
        except Exception as e:
            print(f"Error showing plots: {e}")

# Output stream
def process_batch(df_batch):
    """Process each batch and create visualizations"""
    try:
        create_visualizations(df_batch)
        
        # Store in history for analysis
        for _, row in df_batch.iterrows():
            price_history[row['lot_id']].append({
                'timestamp': row['timestamp'],
                'price': row['price'],
                'occupancy': row['occupancy'],
                'capacity': row['capacity']
            })
            
    except Exception as e:
        print(f"Error processing batch: {e}")

# Set up output
pw.io.python.write_stream(
    priced_stream,
    process_batch,
    autocommit_duration_ms=1000  # Update every 1 second
)

if __name__ == "__main__":
    print("Starting dynamic pricing system...")
    print("Press Ctrl+C to stop")
    
    try:
        # Enable notebook output if in Jupyter
        try:
            output_notebook()
        except:
            pass
            
        # Run the streaming system
        pw.run()
    except KeyboardInterrupt:
        print("\nSystem stopped by user")
    except Exception as e:
        print(f"Error running system: {e}")