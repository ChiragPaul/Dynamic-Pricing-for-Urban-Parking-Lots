# !pip install pathway bokeh pandas numpy

import pandas as pd
import numpy as np
import time
import os
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Bokeh imports for visualization
from bokeh.plotting import figure, show, output_notebook
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.layouts import gridplot, column
from bokeh.io import push_notebook
from bokeh.palettes import Category20

# Try to import Pathway (install if needed)
try:
    import pathway as pw
    PATHWAY_AVAILABLE = True
    print("Pathway is available - will use real-time streaming")
except ImportError:
    PATHWAY_AVAILABLE = False
    print("Pathway not available - will use batch processing simulation")

print("Dynamic Pricing System for Urban Parking Lots")
print("=" * 60)

# Generating the data

def create_dataset():
    """Generate realistic parking lot dataset"""
    np.random.seed(42)

    # Parameters
    num_lots = 14
    num_days = 10  # Reduced for faster demo
    times_per_day = 18  # 8:00 AM to 4:30 PM with 30 min intervals

    # Generate lot IDs and their fixed properties
    lots = []
    for i in range(num_lots):
        lot_data = {
            'lot_id': i + 1,
            'lat': 40.7128 + np.random.normal(0, 0.005),  # Around NYC, closer together
            'lon': -74.0060 + np.random.normal(0, 0.005),
            'capacity': np.random.randint(20, 101)  # 20-100 spaces
        }
        lots.append(lot_data)

    # Generate time series data
    data = []
    from datetime import datetime, timedelta
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
    return df


# Loadingg the dataset
dataset_file = 'dataset.csv'
if os.path.exists(dataset_file):
    try:
        df = pd.read_csv(dataset_file, parse_dates=['timestamp'])
        if 'timestamp' not in df.columns:
            print(f"'{dataset_file}' exists but is missing 'timestamp' column. Regenerating dataset.")
            df = create_dataset()
        else:
            print(f"Dataset loaded from '{dataset_file}': {len(df)} records, {df['lot_id'].nunique()} lots")
            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    except Exception as e:
        print(f"Error loading '{dataset_file}': {e}. Regenerating dataset.")
        df = create_dataset()
else:
    print(f"'{dataset_file}' not found. Creating dataset.")
    df = create_dataset()


# Various Pricing Models

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

def model1_linear(prev_price, occupancy, capacity):
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
    
#Real time simulation graphs

class ParkingPricingSystem:
    def __init__(self, df):
        self.df = df
        self.plot_data = defaultdict(lambda: ColumnDataSource(data=dict(
            timestamp=[], price=[], occupancy_rate=[], queue=[], traffic=[]
        )))
        self.price_history = defaultdict(list)
        self.current_prices = {}

    def process_batch(self, batch_df):
        """Process a batch of data and update prices"""
        results = []

        for idx, row in batch_df.iterrows():
            # Calculate prices using all three models
            base_price = 10.0

            price1 = model1_linear(base_price, row['occupancy'], row['capacity'])
            price2 = model2_demand_based(row)
            price3, message = model3_competitive(row, batch_df)

            # Use Model 3 (competitive) as the final price
            final_price = price3

            self.current_prices[row['lot_id']] = final_price

            # Store result
            result = {
                'lot_id': row['lot_id'],
                'timestamp': row['timestamp'],
                'occupancy': row['occupancy'],
                'capacity': row['capacity'],
                'occupancy_rate': row['occupancy'] / row['capacity'] if row['capacity'] > 0 else 0,
                'queue': row['queue'],
                'traffic': row['traffic'],
                'is_special': row['is_special'],
                'vehicle_type': row['vehicle_type'],
                'price1': price1,
                'price2': price2,
                'price3': price3,
                'final_price': final_price,
                'message': message
            }
            results.append(result)

            # Update price history
            self.price_history[row['lot_id']].append(result)

        return results

    def create_visualizations(self, results):
        """Create Bokeh visualizations"""
        if not results:
            return None

        # Convert results to DataFrame for easier handling
        results_df = pd.DataFrame(results)

        # Create individual plots for each lot
        plots = []
        colors = Category20[20]

        for i, lot_id in enumerate(sorted(results_df['lot_id'].unique())):
            lot_data = results_df[results_df['lot_id'] == lot_id]

            if lot_data.empty:
                continue

            # Create figure
            p = figure(
                title=f"Lot {lot_id} - Dynamic Pricing",
                x_axis_type='datetime',
                width=400,
                height=300,
                tools="pan,wheel_zoom,box_zoom,reset,save"
            )

            # Add hover tool
            hover = HoverTool(tooltips=[
                ('Time', '@timestamp{%F %T}'),
                ('Price', '$@final_price{0.00}'),
                ('Occupancy', '@occupancy_rate{0.0%}'),
                ('Queue', '@queue'),
                ('Traffic', '@traffic{0.00}')
            ], formatters={'@timestamp': 'datetime'})
            p.add_tools(hover)

            # Convert timestamp to datetime for plotting
            timestamps = pd.to_datetime(lot_data['timestamp'])

            # Price line
            p.line(timestamps, lot_data['final_price'], line_width=3,
                   color=colors[i % len(colors)], legend_label='Price ($)')

            # Occupancy rate (scaled to price range for visibility)
            max_price = lot_data['final_price'].max()
            occupancy_scaled = lot_data['occupancy_rate'] * max_price
            p.line(timestamps, occupancy_scaled, line_width=2,
                   color='red', alpha=0.7, legend_label='Occupancy (scaled)')

            # Queue indicators
            queue_mask = lot_data['queue'] > 0
            if queue_mask.any():
                p.circle(timestamps[queue_mask], lot_data['final_price'][queue_mask],
                        size=8, color='orange', alpha=0.7, legend_label='Queue > 0')

            p.xaxis.axis_label = "Time"
            p.yaxis.axis_label = "Price ($)"
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"

            plots.append(p)

        # Create summary plot
        summary_plot = self.create_summary_plot(results_df)
        plots.append(summary_plot)

        return gridplot(plots, ncols=2)

    def create_summary_plot(self, results_df):
        """Create a summary plot showing all lots"""
        p = figure(
            title="All Lots - Price Comparison",
            x_axis_type='datetime',
            width=800,
            height=400,
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        colors = Category20[20]

        for i, lot_id in enumerate(sorted(results_df['lot_id'].unique())):
            lot_data = results_df[results_df['lot_id'] == lot_id]

            if lot_data.empty:
                continue

            timestamps = pd.to_datetime(lot_data['timestamp'])

            p.line(timestamps, lot_data['final_price'], line_width=2,
                   color=colors[i % len(colors)], legend_label=f'Lot {lot_id}')

        p.xaxis.axis_label = "Time"
        p.yaxis.axis_label = "Price ($)"
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"

        return p

    def run_simulation(self, delay=1.0, max_batches=10):
        """Run the pricing simulation"""
        print(f"Starting parking pricing simulation...")
        print(f"Processing {max_batches} time batches with {delay}s delay")
        print("=" * 60)

        # Enable notebook output
        output_notebook()

        batch_count = 0

        # Group by timestamp to simulate real-time processing
        for timestamp, batch in self.df.groupby('timestamp'):
            if batch_count >= max_batches:
                break

            print(f"\nProcessing batch {batch_count + 1}/{max_batches}")
            print(f"Timestamp: {timestamp}")
            print("-" * 40)

            # Process the batch
            results = self.process_batch(batch)

            # Display results
            for result in sorted(results, key=lambda x: x['final_price']):
                print(f"Lot {result['lot_id']:2d}: ${result['final_price']:6.2f} "
                      f"(Occ: {result['occupancy_rate']:.0%}, Q: {result['queue']}) "
                      f"- {result['message']}")

            # Create and show visualization
            plot = self.create_visualizations(results)
            if plot:
                show(plot)

            # Add delay
            time.sleep(delay)
            batch_count += 1

        print(f"\nSimulation completed!")
        self.print_summary()

    def print_summary(self):
        """Print simulation summary"""
        print("\n" + "=" * 60)
        print("SIMULATION SUMMARY")
        print("=" * 60)

        if not self.price_history:
            print("No pricing data available")
            return

        for lot_id in sorted(self.price_history.keys()):
            history = self.price_history[lot_id]
            if not history:
                continue

            prices = [h['final_price'] for h in history]
            occupancy_rates = [h['occupancy_rate'] for h in history]

            print(f"Lot {lot_id:2d}: "
                  f"Avg Price: ${np.mean(prices):5.2f}, "
                  f"Range: ${min(prices):5.2f}-${max(prices):5.2f}, "
                  f"Avg Occupancy: {np.mean(occupancy_rates):4.0%}")

def main():
    """Main function to run the parking pricing system"""

    # Create the pricing system
    pricing_system = ParkingPricingSystem(df)

    # Run the simulation
    pricing_system.run_simulation(delay=0.5, max_batches=5)

    # Additional analysis
    print("\n" + "=" * 60)
    print("ADDITIONAL ANALYSIS")
    print("=" * 60)

    # Model comparison on sample data
    sample_data = df.head(20)
    print("\nModel Comparison (Sample Data):")
    print("-" * 40)

    for idx, row in sample_data.iterrows():
        price1 = model1_linear(10.0, row['occupancy'], row['capacity'])
        price2 = model2_demand_based(row)
        price3, message = model3_competitive(row, sample_data)

        print(f"Lot {row['lot_id']:2d}: "
              f"Model1=${price1:5.2f}, "
              f"Model2=${price2:5.2f}, "
              f"Model3=${price3:5.2f} "
              f"({message})")

    print("\n" + "=" * 60)
    print("SYSTEM COMPLETED SUCCESSFULLY!")
    print("=" * 60)


main()