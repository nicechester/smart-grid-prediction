# ⚡ Smart Grid ML - Electricity Price Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.14.0-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A machine learning system for predicting electricity prices in California's smart grid using real-time data from multiple sources including power plants, weather, grid demand, and environmental factors.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Data Sources](#data-sources)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Model Details](#model-details)
- [Training](#training)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Smart Grid ML predicts electricity prices for California's power grid (CAISO) using a deep neural network trained on:
- **Tier 2 Real Data**: Power plants, weather, historical prices, seismic activity, grid topology
- **Synthetic Fallback**: Generated realistic data when APIs are unavailable

### Key Capabilities

- 🔮 **Price Prediction**: Real-time electricity price forecasting
- 🌍 **Location-Aware**: Predictions for specific California regions, counties, and cities
- 📊 **Multi-Source Integration**: Combines 5+ data sources
- 🚀 **Production Ready**: Docker deployment with REST API
- 🔄 **Retrainable**: Automated model retraining pipeline

---

## 📖 Repository

- **GitHub Repository**: [Smart Grid Prediction](https://github.com/nicechester/smart-grid-prediction)

---

## ✨ Features

### Core Features

- **Deep Learning Model**: TensorFlow neural network (128→64→32→16→1 architecture)
- **Real-Time Predictions**: Sub-second inference via REST API
- **Historical Analysis**: Train on EIA historical demand/price data
- **Weather Integration**: NOAA weather forecasts
- **Geospatial Data**: Power plant locations, transmission lines via OpenStreetMap
- **Risk Factors**: Earthquake data from USGS
- **Location Profiles**: 13 major California cities with unique demand characteristics

### Data Pipeline (Tier 2)

1. **Power Plants** - CartoDB Global Power Plant Database (~30,000 plants)
2. **Grid Topology** - OSMnx transmission lines from OpenStreetMap
3. **Weather** - NOAA forecasts (temperature, wind, cloud cover)
4. **Historical Prices** - EIA electricity demand/generation data
5. **Seismic Activity** - USGS earthquake data

---

## 🏗️ Updated Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     CLIENT (Web Browser)                    │
└──────────────────────────┬──────────────────────────────────┘
                           │ HTTP/REST
┌──────────────────────────▼──────────────────────────────────┐
│                    FLASK WEB SERVICE                        │
│                       (app.py)                              │
│  Routes: /predict, /tier2-data, /health, /model-info        │
└──────────────────────────┬──────────────────────────────────┘
                           │
        ┌──────────────────┼──────────────────┐
        │                  │                  │
┌───────▼────────┐  ┌──────▼──────┐  ┌───────▼────────┐
│  PricePredictor│  │Tier2Pipeline│  │   Locations    │
│   (main.py)    │  │(tier2_*.py) │  │ (locations.py) │
│                │  │             │  │                │
│ • Model        │  │ • CartoDB   │  │ • CA Regions   │
│ • Training     │  │ • NOAA      │  │ • Cities       │
│ • Prediction   │  │ • EIA       │  │ • Counties     │
│ • Save/Load    │  │ • USGS      │  │ • Profiles     │
│                │  │ • PowerDB   │  │                │
└────────────────┘  │             │  └────────────────┘
                    └─────────────┘
                           │
                    ┌──────▼──────┐
                    │External APIs│
                    │             │
                    │ • CartoDB   │
                    │ • NOAA      │
                    │ • EIA       │
                    │ • USGS      │
                    │ • OSM       │
                    └─────────────┘
```

### Component Flow

```
Training Flow:
train.py → Tier2Pipeline → External APIs → Data Processing → 
PricePredictor.train() → Save Model → models/

Prediction Flow:
app.py → Load Model → /predict endpoint → 
PricePredictor.predict() → JSON Response
```

---

## 📊 Updated Data Sources

### 1. CartoDB - Power Plant Database
- **Source**: World Resources Institute (WRI)
- **Data**: Global Power Plant Database
- **Access**: CartoDB SQL API
- **Coverage**: ~30,000 power plants worldwide
- **CA Plants**: ~500+ facilities
- **Fields**: Name, fuel type, capacity (MW), coordinates

### 2. NOAA - Weather Data
- **Source**: National Weather Service API
- **Endpoint**: `api.weather.gov`
- **Data**: 24-hour forecasts
- **Fields**: Temperature, wind speed, conditions
- **Update**: Hourly
- **No API Key Required**

### 3. EIA - Energy Information Administration
- **Source**: U.S. Energy Information Administration
- **API**: EIA Open Data v2
- **Data**: Hourly electricity demand, net generation
- **Region**: CAISO (California ISO)
- **History**: Up to 5,000 records
- **Requires**: API Key (free)

### 4. USGS - Earthquake Data
- **Source**: U.S. Geological Survey
- **Endpoint**: `earthquake.usgs.gov`
- **Data**: Recent earthquakes (7 days)
- **Fields**: Magnitude, depth, location, time
- **Update**: Real-time
- **No API Key Required**

### 5. OpenStreetMap - Grid Topology
- **Source**: OpenStreetMap via OSMnx
- **Library**: `osmnx` Python package
- **Data**: High-voltage transmission lines
- **Tag**: `power=line`
- **Optional**: Large queries may timeout

---

## 🚀 Installation

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for containerized deployment)
- 4GB+ RAM
- EIA API Key (optional, for real data)

### Local Installation

```bash
# Clone repository
git clone https://github.com/nicechester/smart-grid-prediction.git
cd smart-grid-ml

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
EIA_API_KEY=your_api_key_here
EOF

# Train model
python train.py

# Start web service
python app.py
```

### Docker Installation

```bash
# Clone repository
git clone https://github.com/nicechester/smart-grid-prediction.git
cd smart-grid-ml

# Create .env file
echo "EIA_API_KEY=your_api_key_here" > .env

# Build and run
docker-compose up
```

---

## 📖 Usage

### Web Interface

1. **Start the service**:
   ```bash
   python app.py
   # Or with Docker:
   docker-compose up smart_grid_api
   ```

2. **Open browser**: `http://localhost:8000`

3. **Features**:
   - Click "Get Prediction" for instant price forecast
   - "Data Sources" shows Tier 2 data status
   - "Model Info" displays model metrics
   - "Auto Refresh" enables 10-second updates

### REST API

#### Get Price Prediction

```bash
curl http://localhost:8000/predict
```

**Response**:
```json
{
  "predicted_price": 45.23,
  "price_level": "MEDIUM",
  "description": "Normal grid conditions",
  "confidence": "High",
  "timestamp": "2025-10-23T04:20:00",
  "location": "California",
  "model_version": "1.0.0"
}
```

#### Get Tier 2 Data Status

```bash
curl http://localhost:8000/tier2-data
```

**Response**:
```json
{
  "power_plants": {
    "count": 487,
    "available": true
  },
  "transmission_lines": {
    "count": 1234,
    "available": true
  },
  "weather": {
    "temperature": 22.5,
    "wind_speed": 4.2,
    "available": true
  },
  "prices": {
    "count": 5000,
    "available": true
  },
  "earthquakes": {
    "count": 42,
    "available": true
  }
}
```

#### Health Check

```bash
curl http://localhost:8000/health
```

#### Model Info

```bash
curl http://localhost:8000/model-info
```

---

## 📁 Project Structure

```
smart-grid-ml/
│
├── main.py                 # Core ML library (model, config)
├── train.py                # Training pipeline
├── app.py                  # Flask web service
├── tier2_pipeline.py       # Data fetching pipeline
├── locations.py            # CA locations database
│
├── requirements.txt        # Python dependencies
├── Dockerfile             # Container definition
├── docker-compose.yml     # Multi-service orchestration
├── .env                   # Environment variables (API keys)
│
├── models/                # Trained models (generated)
│   ├── price_model.keras     # Keras model
│   ├── scaler.pkl         # Feature scaler
│   ├── features.json      # Feature names
│   └── training_metadata.json
│
├── data/                  # Data cache (generated)
├── tier2_data/           # Tier 2 data cache (generated)
│
├── index.html            # Web UI
├── README.md             # This file
└── training.log          # Training logs (generated)
```

### Key Files

| File | Purpose |
|------|---------|
| `main.py` | Core ML library with `PricePredictor` class |
| `train.py` | Model training script with Tier 2 integration |
| `app.py` | Flask REST API server |
| `tier2_pipeline.py` | Data fetching from 5 external sources |
| `locations.py` | California location database (13 cities, 10 counties, 3 regions) |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Single container image for all services |
| `docker-compose.yml` | Orchestrates trainer + API services |

---

## ⚙️ Configuration

### Environment Variables (.env)

```bash
# EIA API Key (get from https://www.eia.gov/opendata/register.php)
EIA_API_KEY=your_api_key_here

# Optional: Override defaults
MODEL_DIR=models
DATA_DIR=data
```

### Training Configuration

Edit `train.py` → `TrainingConfig`:

```python
class TrainingConfig:
    EPOCHS = 50              # Training epochs
    BATCH_SIZE = 32          # Batch size
    TEST_SIZE = 0.2          # Test split ratio
    MIN_SAMPLES = 500        # Minimum samples to train
    VALIDATION_SPLIT = 0.2   # Validation split
    RANDOM_STATE = 42        # Reproducibility
```

### Model Configuration

Edit `main.py` → `Config`:

```python
class Config:
    DATA_DIR = "data"
    MODEL_DIR = "models"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    EPOCHS = 50
    BATCH_SIZE = 32
```

---

## 🤖 Model Details

### Architecture

```
Input Layer (N features)
    ↓
Dense(128, relu) + Dropout(0.2)
    ↓
Dense(64, relu) + BatchNorm + Dropout(0.2)
    ↓
Dense(32, relu) + Dropout(0.1)
    ↓
Dense(16, relu)
    ↓
Dense(1) → Price Output
```

### Features (Example)

**Base Features**:
- `temperature` - Current temperature (°C)
- `cloud_cover` - Cloud coverage (0-1)
- `wind_speed` - Wind speed (m/s)
- `solar_mw` - Solar generation (MW)
- `wind_mw` - Wind generation (MW)
- `total_demand` - Grid demand (MW)
- `renewable_pct` - Renewable percentage
- `imbalance` - Supply/demand ratio
- `grid_stress` - Grid stress level (0-1)
- `wildfire_risk` - Wildfire risk (0-1)
- `hour` - Hour of day (0-23)
- `month` - Month (1-12)
- `is_weekend` - Weekend flag (0/1)

**Lag Features** (added during training):
- `temperature_lag_1, _3, _6, _12` - Historical temperature
- `solar_mw_lag_1, _3, _6, _12` - Historical solar
- `wind_mw_lag_1, _3, _6, _12` - Historical wind
- `price_lag_1, _3, _6, _12` - Historical price

**Total**: ~50+ features

### Performance Metrics

Typical performance on synthetic data:
- **MAE**: 3-5 €/MWh
- **MAPE**: 8-12%
- **R²**: 0.85-0.92

Real EIA data performance varies based on data quality.

---

## 🎓 Training

### Quick Start

```bash
# Train with default settings (50 epochs, Tier 2 data)
python train.py

# Train with custom epochs
python train.py --epochs 100

# Train with synthetic data only (skip Tier 2)
python train.py --no-tier2 --synthetic-days 365

# Train with custom batch size
python train.py --batch-size 64 --epochs 50
```

### Training Options

```bash
python train.py --help
```

**Options**:
- `--epochs INT` - Number of training epochs (default: 50)
- `--batch-size INT` - Batch size (default: 32)
- `--synthetic-days INT` - Synthetic data days if Tier 2 fails (default: 365)
- `--no-tier2` - Skip Tier 2, use synthetic only

### Training Pipeline

1. **Initialize Config** - Load environment variables
2. **Build Training Data**:
   - Try Tier 2 data sources (CartoDB, EIA, NOAA, USGS, OSMnx)
   - Fallback to synthetic data if unavailable
3. **Feature Engineering** - Add lag features (1, 3, 6, 12 hours)
4. **Data Split** - 80% train, 20% test
5. **Build Model** - Create neural network
6. **Train** - Fit model with validation
7. **Evaluate** - Calculate MAE, MAPE on test set
8. **Save** - Export model, scaler, feature names, metadata

### Outputs

After training:
```
models/
├── price_model.keras        # Keras model
├── scaler.pkl               # StandardScaler
├── features.json            # Feature names list
└── training_metadata.json   # Training info + metrics
```

### Logs

Training logs saved to `training.log`:
```
2025-10-23 04:20:00 - INFO - BUILDING TRAINING DATA
2025-10-23 04:20:01 - INFO - ✓ Loaded 5000 real EIA records
2025-10-23 04:20:02 - INFO - TRAINING MODEL
2025-10-23 04:20:10 - INFO - ✓ Test MAE: 4.23 €/MWh
```

---

## 🐳 Docker Deployment

### Architecture

The system uses a **single Dockerfile** with **multiple services** in docker-compose:

```yaml
services:
  smart_grid_trainer:    # One-time training
    command: python train.py
    
  smart_grid_api:        # Web service
    command: python app.py
    ports:
      - "8000:8000"
```

### Commands

```bash
# Build image (done once or when dependencies change)
docker-compose build

# Train model
docker-compose up smart_grid_trainer

# Start API service
docker-compose up smart_grid_api

# Run both (train then serve)
docker-compose up

# Run in background
docker-compose up -d smart_grid_api

# View logs
docker-compose logs -f smart_grid_api

# Stop all
docker-compose down

# Rebuild without cache
docker-compose build --no-cache
```

### Volume Mounts

Models and data persist via volumes:
```yaml
volumes:
  - ./models:/app/models      # Trained models
  - ./data:/app/data          # Data cache
  - ./tier2_data:/app/tier2_data  # Tier 2 cache
```

### Environment Variables

Pass via docker-compose:
```yaml
environment:
  - EIA_API_KEY=${EIA_API_KEY}
```

Or via `.env` file (automatically loaded).

### Production Deployment

```bash
# Build production image
docker-compose build

# Run API only (assumes model already trained)
docker-compose up -d smart_grid_api

# Check health
curl http://localhost:8000/health
```

---

## 👨‍💻 Development

### Setup Development Environment

```bash
# Clone repo
git clone https://github.com/nicechester/smart-grid-prediction.git
cd smart-grid-ml

# Create venv
python -m venv venv
source venv/bin/activate

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8

# Setup pre-commit hooks (optional)
pre-commit install
```

### Development Workflow

1. **Make code changes**
2. **Train locally**: `python train.py --epochs 10`
3. **Test API**: `python app.py`
4. **Test in Docker**: `docker-compose up --build`

### Code Style

```bash
# Format code
black *.py

# Lint
flake8 *.py

# Type check
mypy *.py
```

### Testing

```bash
# Unit tests
pytest tests/

# Integration test
python -m pytest tests/test_integration.py

# API test
curl http://localhost:8000/health
```

---

## 🐛 Troubleshooting

### Common Issues

#### 1. EIA API Returns No Data

**Problem**: `No data found from EIA`

**Solutions**:
- Check API key is valid: `echo $EIA_API_KEY`
- Verify key at: https://www.eia.gov/opendata/
- Training will fallback to synthetic data automatically

#### 2. OSMnx Timeout

**Problem**: `Transmission lines fetch failed: timeout`

**Solutions**:
- This is optional data - system works without it
- Reduce query area in `locations.py`
- Skip with: `python train.py --no-tier2`

#### 3. Model Not Loading

**Problem**: `Model not loaded` in web UI

**Solutions**:
```bash
# Train model first
python train.py

# Verify model exists
ls -lh models/

# Check file permissions
chmod 644 models/*
```

#### 4. Docker Build Slow

**Problem**: Docker rebuild takes forever

**Solutions**:
```bash
# Use image caching
docker-compose build  # First time (slow)
docker-compose build  # Second time (fast with cache)

# For development, use volumes (no rebuild needed):
# Edit docker-compose.yml → add volume: - .:/app
```

#### 5. Port Already in Use

**Problem**: `Port 8000 already in use`

**Solutions**:
```bash
# Find process
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
# Edit docker-compose.yml → ports: "8001:8000"
```

#### 6. Timestamp Error During Training

**Problem**: `TypeError: float() argument must be a string or a real number, not 'Timestamp'`

**Solutions**:
- Fixed in latest `train.py` - datetime columns dropped before training
- Update to latest code
- Verify: `git pull origin main`

### Debug Mode

Enable verbose logging:
```bash
# Set in code
logging.basicConfig(level=logging.DEBUG)

# Or environment variable
export LOG_LEVEL=DEBUG
python train.py
```

### Get Help

1. Check logs: `tail -f training.log`
2. Check Docker logs: `docker-compose logs -f`
3. Verify API keys: `cat .env`
4. Test external APIs:
   ```bash
   curl "https://api.weather.gov/points/34.05,-118.24"
   ```

---

## 📈 Future Enhancements

### Planned Features

- [ ] Real-time streaming predictions
- [ ] Multi-region support (beyond California)
- [ ] Advanced wildfire risk integration (NASA FIRMS)
- [ ] Grid congestion modeling
- [ ] Time-series forecasting (LSTM/Transformer)
- [ ] Model explainability (SHAP values)
- [ ] A/B testing framework
- [ ] Kubernetes deployment
- [ ] GraphQL API
- [ ] Mobile app

### Data Source Expansions

- [ ] CAISO real-time LMP prices
- [ ] PG&E demand data
- [ ] Solar irradiance (NREL)
- [ ] Electric vehicle charging patterns
- [ ] Battery storage levels
- [ ] Natural gas prices (Henry Hub)

---

## 🤝 Contributing

Contributions welcome! Please follow these steps:

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open Pull Request**

### Contribution Guidelines

- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Keep commits atomic and descriptive

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Data Sources

- **CartoDB/WRI** - Global Power Plant Database
- **NOAA** - Weather forecasts
- **EIA** - Energy data
- **USGS** - Seismic data
- **OpenStreetMap** - Grid topology

### Technologies

- **TensorFlow** - Deep learning framework
- **Flask** - Web framework
- **OSMnx** - Geospatial analysis
- **Docker** - Containerization
- **scikit-learn** - ML utilities

---

## 📞 Contact

- **Project Link**: https://github.com/nicechester/smart-grid-prediction
- **Issues**: https://github.com/nicechester/smart-grid-prediction/issues

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/nicechester/smart-grid-prediction)
![GitHub forks](https://img.shields.io/github/forks/nicechester/smart-grid-prediction)
![GitHub issues](https://img.shields.io/github/issues/nicechester/smart-grid-prediction)

---

**Built with ❤️ for a sustainable energy future**
