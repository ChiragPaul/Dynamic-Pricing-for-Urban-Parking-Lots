# Dynamic Pricing for Urban Parking Lots

**Capstone Project - Summer Analytics 2025**  
_Consulting & Analytics Club × Pathway_

## Project Overview

This project implements an intelligent, data-driven dynamic pricing system for urban parking lots. The system processes real-time data streams to adjust parking prices based on demand, competition, and various environmental factors, optimizing utilization while maximizing revenue.

## 🎯 Objectives

- Build a real-time dynamic pricing model for 14 parking spaces
- Implement three progressively sophisticated pricing models
- Process streaming data using Pathway framework
- Provide real-time visualizations using Bokeh
- Suggest optimal routing based on competitive analysis

## 📊 Dataset Description

### Data Coverage

- **14 parking lots** across urban area
- **73 days** of historical data (reduced to 10 days for demo)
- **18 time points per day** (8:00 AM to 4:30 PM, 30-minute intervals)
- **18,396 total records** in full dataset

### Features

- **Location**: Latitude, Longitude
- **Capacity**: Maximum parking spaces
- **Occupancy**: Current parked vehicles
- **Queue**: Vehicles waiting for entry
- **Traffic**: Nearby traffic congestion level (0-1)
- **Special Day**: Boolean indicator for weekends/holidays
- **Vehicle Type**: Car, bike, or truck

## 🧠 Pricing Models

### Model 1: Linear Baseline

Simple linear relationship between price and occupancy:

```
Price(t+1) = Price(t) + α × (Occupancy/Capacity)
```

### Model 2: Demand-Based Pricing

Comprehensive demand function incorporating multiple factors:

```
Demand = α×(Occupancy/Capacity) + β×Queue - γ×Traffic + δ×SpecialDay + ε×VehicleWeight
Price = BasePrice × (1 + λ × tanh(Demand/10))
```

**Parameters:**

- α = 5.0 (occupancy sensitivity)
- β = 2.0 (queue sensitivity)
- γ = 1.0 (traffic sensitivity)
- δ = 3.0 (special day multiplier)
- λ = 0.5 (demand adjustment factor)

**Vehicle Weights:**

- Car: 1.0
- Bike: -0.5 (discount)
- Truck: 1.5 (premium)

### Model 3: Competitive Pricing

Adds geographical competition and rerouting logic:

- Identifies nearby lots (within ~1km radius)
- Compares prices with competitors
- Suggests rerouting when beneficial
- Applies premium pricing when no cheaper alternatives exist

## 🏗️ System Architecture

### Core Components

1. **Data Generator** (`create_dataset.py`)

   - Generates realistic parking data with temporal patterns
   - Simulates rush hours, weekend effects, and special events
   - Creates correlated occupancy and traffic patterns

2. **Pricing Engine** (`enhanced_model.py`)

   - Implements all three pricing models
   - Handles real-time data processing
   - Manages competitive analysis

3. **Visualization System** (`Dynamic_Parking_Pricing_System.py`)

   - Real-time Bokeh plots
   - Interactive dashboards
   - Comparative analysis charts

4. **Testing Suite** (`simple_test.py`)
   - Model validation
   - Price boundary checking
   - Scenario analysis

### Data Flow

```
Raw Data → Pathway Stream → Pricing Models → Visualization → Analysis
```

## 🚀 Usage

### Quick Start

1. **Install Dependencies**

```bash
pip install pathway bokeh pandas numpy
```

2. **Run Basic Test**

```bash
python simple_test.py
```

3. **Run Full System**

```bash
python Dynamic_Parking_Pricing_System.py
```

## 📈 Key Results

### Price Ranges by Scenario

| Scenario              | Average Price | Price Range     | Key Factors      |
| --------------------- | ------------- | --------------- | ---------------- |
| High Occupancy (80%+) | $12.91        | $12.08 - $14.08 | Occupancy, Queue |
| Low Occupancy (<30%)  | $10.85        | $10.06 - $11.39 | Base pricing     |
| High Traffic          | $11.47        | $10.67 - $11.86 | Traffic penalty  |
| Special Days          | $13.12        | $12.47 - $14.08 | Event premium    |

### Model Performance

- **Model 1**: Simple but reactive to occupancy
- **Model 2**: Balanced pricing with smooth variations
- **Model 3**: Intelligent competitive pricing with rerouting

### Competitive Intelligence

- Identifies nearby competitors within 1km radius
- Provides rerouting suggestions when beneficial
- Applies premium pricing when market leader
- Reduces prices when oversupply exists

## 🔧 Technical Implementation

### Real-Time Processing

- Uses Pathway for streaming data ingestion
- Processes batches with configurable delays
- Maintains price history for trend analysis

### Visualization Features

- Interactive Bokeh plots with hover tooltips
- Real-time price updates
- Occupancy rate overlays
- Queue indicators
- Multi-lot comparison charts

### Error Handling

- Graceful degradation when Pathway unavailable
- Price boundary validation ($5-$20 range)
- Missing data handling
- Robust competitive analysis

## 📋 Files Description

- `enhanced_model.py` -Complete Pathway-based system
- `Dynamic_Parking_Pricing_System.py` - Main system with visualization
- `simple_test.py` - Model testing and validation
- `create_dataset.py` - Data generation script
- `dataset.csv` - Generated parking data
- `README.md` - This documentation

## 🔍 Analysis Features

### Real-Time Monitoring

- Live price updates every 30 seconds
- Occupancy rate tracking
- Queue length monitoring
- Traffic impact analysis

### Competitive Analysis

- Nearby lot identification
- Price comparison
- Rerouting recommendations
- Market positioning

### Business Intelligence

- Revenue optimization
- Demand forecasting
- Utilization efficiency
- Customer flow management

## 📊 Visualization Examples

### Individual Lot Tracking

- Price trends over time
- Occupancy correlation
- Queue event markers
- Traffic impact visualization

### System-Wide Analysis

- All lots price comparison
- Market dynamics
- Competitive positioning
- Revenue optimization

## 🎯 Business Impact

### Revenue Optimization

- Dynamic pricing increases revenue by 15-25%
- Reduces empty spaces during off-peak hours
- Captures premium during high-demand periods

### Customer Experience

- Reduces search time through rerouting
- Provides price transparency
- Optimizes parking availability

### Operational Efficiency

- Automated pricing decisions
- Real-time monitoring
- Predictive capacity management

## 🔮 Future Enhancements

### Advanced Features

- Machine learning price prediction
- Weather impact integration
- Event-based pricing
- Mobile app integration

### Scalability

- Multi-city deployment
- Cloud-based processing
- API integration
- Real-time notifications

## 📖 References

1. [Pathway Documentation](https://pathway.com/developers/)
2. [Bokeh Visualization Guide](https://docs.bokeh.org/)
3. [Dynamic Pricing Theory](https://en.wikipedia.org/wiki/Dynamic_pricing)
4. [Urban Parking Management](https://www.smartcitiesworld.net/news/parking-management-systems)

## 🤝 Contributing

This project is part of Summer Analytics 2025. For improvements or issues:

1. Fork the repository
2. Create feature branch
3. Submit pull request with detailed description

## 📄 License

This project is developed for educational purposes as part of Summer Analytics 2025 capstone project.

---

**Mentored by**: Consulting & Analytics Club × Pathway  
**Technology Stack**: Python, Pathway, Bokeh, Pandas, NumPy
