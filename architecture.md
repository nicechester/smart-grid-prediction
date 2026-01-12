# 🏗️ Architecture: Geolocation-Based Price Prediction

This document provides a detailed technical overview of the Smart Grid ML system architecture for predicting electricity prices based on geographic coordinates.

---

## System Overview

The system predicts California ISO (CAISO) Locational Marginal Prices (LMP) for any latitude/longitude coordinate in California. It uses a deep learning model trained on historical prices from 4,300+ CAISO nodes combined with weather, power plant proximity, and temporal features.

```mermaid
flowchart TB
    subgraph External["External Data Sources"]
        CAISO["CAISO OASIS API<br/>Historical LMP Prices"]
        NOAA["NOAA GHCND<br/>Weather Stations"]
        CartoDB["CartoDB<br/>Power Plants"]
        PriceMap["caiso-price-map.json<br/>Node Coordinates"]
    end
    
    subgraph Pipeline["Data Pipeline"]
        Download["download_geo.py<br/>Fetch Prices"]
        Extract["caiso_nodes.py<br/>Extract CA Nodes"]
        Features["geo_features.py<br/>Build Features"]
    end
    
    subgraph Training["Model Training"]
        TrainScript["train_geo.py"]
        TFModel["TensorFlow Model<br/>Dense NN"]
        Scaler["StandardScaler"]
    end
    
    subgraph Serving["Production Serving"]
        Flask["app_geo.py<br/>Flask API"]
        Predict["GeoPricePredictor"]
        Login["Session Auth"]
    end
    
    subgraph Clients["Clients"]
        Browser["Web Browser<br/>Google Maps UI"]
        REST["REST API<br/>Programmatic Access"]
    end
    
    CAISO --> Download
    PriceMap --> Extract
    Extract --> Download
    NOAA --> Features
    CartoDB --> Features
    Download --> Features
    
    Features --> TrainScript
    TrainScript --> TFModel
    TrainScript --> Scaler
    
    TFModel --> Flask
    Scaler --> Flask
    Flask --> Predict
    Flask --> Login
    
    Browser --> Flask
    REST --> Flask
```

---

## Core Components

### 1. Data Layer

#### CAISO Nodes (`caiso_nodes.py`)

Manages the 4,300+ California ISO pricing nodes extracted from `caiso-price-map.json`.

```mermaid
classDiagram
    class CAISONodes {
        +Dict nodes
        +load_from_file(path)
        +get_node(node_id) Dict
        +find_nearest_node(lat, lon) Tuple
        +find_nodes_within_radius(lat, lon, km) List
        +get_nodes_by_area(area) List
    }
    
    class NodeData {
        +str node_id
        +float latitude
        +float longitude
        +str node_type
        +str area
        +float day_ahead_price
        +float congestion
        +float loss
    }
    
    CAISONodes "1" --> "*" NodeData
```

**Key Functions:**
- `get_california_nodes()` - Load all CA nodes from JSON
- `find_nearest_node(lat, lon)` - Find closest pricing node to coordinates
- `export_california_nodes()` - Generate `caiso_nodes_california.json`

#### Price Downloader (`download_geo.py`)

Fetches historical LMP prices from CAISO OASIS API.

```mermaid
sequenceDiagram
    participant Script as download_geo.py
    participant CAISO as CAISO OASIS API
    participant Storage as geo_prices.pkl
    
    Script->>Script: Load CA nodes list
    Script->>Script: Select sample nodes
    
    loop For each node
        Script->>Script: Log timing start
        loop For each 29-day chunk
            Script->>CAISO: GET SingleZip (DAM_LMP)
            CAISO-->>Script: ZIP with CSV
            Script->>Script: Parse CSV, filter LMP_PRC
        end
        Script->>Script: Log elapsed/ETA
        Script->>Storage: Checkpoint save
    end
    
    Script->>Storage: Save geo_prices.pkl
```

**CAISO API Details:**
- Endpoint: `http://oasis.caiso.com/oasisapi/SingleZip`
- Query Type: `PRC_LMP`
- Market: `DAM` (Day-Ahead Market)
- Format: ZIP containing CSV
- Rate Limit: 2 second delay between requests

### 2. Feature Engineering Layer (`geo_features.py`)

Transforms raw data into ML-ready features.

```mermaid
flowchart LR
    subgraph Inputs
        Prices["geo_prices.pkl<br/>Price History"]
        Weather["stations.json<br/>NOAA Data"]
        Plants["power_plants.pkl<br/>Generation"]
    end
    
    subgraph Processing
        Pivot["Pivot Weather<br/>Long → Wide"]
        Interp["Interpolate Weather<br/>IDW from Stations"]
        Spatial["Spatial Features<br/>Distance, Proximity"]
        Temporal["Time Features<br/>Hour, Month Encoding"]
        Lag["Lag Features<br/>1, 3, 6, 12 hours"]
    end
    
    subgraph Output
        Training["geo_training.pkl<br/>~50+ Features"]
    end
    
    Prices --> Pivot
    Weather --> Interp
    Pivot --> Interp
    Interp --> Spatial
    Plants --> Spatial
    Spatial --> Temporal
    Temporal --> Lag
    Lag --> Training
```

**Feature Categories:**

| Category | Features | Description |
|----------|----------|-------------|
| **Spatial** | `latitude`, `longitude`, `distance_to_nearest_plant_km`, `plants_within_50km`, `total_capacity_within_50km`, `avg_price_neighbors_25km` | Location-based features |
| **Temporal** | `hour_sin`, `hour_cos`, `month_sin`, `month_cos`, `day_of_week`, `is_weekend` | Cyclical time encoding |
| **Weather** | `temperature`, `wind_speed`, `precipitation`, `cloud_cover` | Interpolated from nearby stations |
| **Generation** | `solar_mw`, `wind_mw`, `total_demand`, `renewable_pct` | Derived from weather/time |
| **Lag** | `price_lag_1`, `price_lag_3`, `price_lag_6`, `price_lag_12`, etc. | Historical values |

### 3. Model Layer (`train_geo.py`)

#### Neural Network Architecture

```mermaid
graph TD
    subgraph Input["Input Layer"]
        I["Features<br/>(N dimensions)"]
    end
    
    subgraph Hidden["Hidden Layers"]
        D1["Dense(128)<br/>ReLU"]
        BN1["BatchNorm"]
        DO1["Dropout(0.2)"]
        
        D2["Dense(64)<br/>ReLU"]
        BN2["BatchNorm"]
        DO2["Dropout(0.2)"]
        
        D3["Dense(32)<br/>ReLU"]
        BN3["BatchNorm"]
        DO3["Dropout(0.2)"]
    end
    
    subgraph Output["Output Layer"]
        O["Dense(1)<br/>Linear"]
        P["Price $/MWh"]
    end
    
    I --> D1 --> BN1 --> DO1
    DO1 --> D2 --> BN2 --> DO2
    DO2 --> D3 --> BN3 --> DO3
    DO3 --> O --> P
    
    style I fill:#3b82f6,color:#fff
    style D1 fill:#8b5cf6,color:#fff
    style D2 fill:#7c3aed,color:#fff
    style D3 fill:#6d28d9,color:#fff
    style O fill:#22c55e,color:#fff
    style P fill:#10b981,color:#fff
```

#### Training Pipeline

```mermaid
flowchart TD
    Load["Load geo_training.pkl"]
    Split["Train/Test Split<br/>80/20"]
    Scale["Fit StandardScaler<br/>on Train Only"]
    Build["Build NN Model"]
    
    subgraph Training["Training Loop"]
        Fit["model.fit()"]
        ES["Early Stopping<br/>patience=10"]
        LR["LR Reduction<br/>on Plateau"]
    end
    
    Eval["Evaluate on Test"]
    Save["Save Model Artifacts"]
    
    Load --> Split --> Scale --> Build --> Fit
    Fit --> ES
    Fit --> LR
    ES --> Eval
    Eval --> Save
    
    Save --> M1["geo_model.keras"]
    Save --> M2["geo_scaler.pkl"]
    Save --> M3["geo_features.json"]
    Save --> M4["geo_metadata.json"]
```

#### Model Configuration

```python
class GeoModelConfig:
    # Training
    EPOCHS = 100
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    EARLY_STOPPING_PATIENCE = 10
    
    # Architecture
    HIDDEN_LAYERS = [128, 64, 32]
    DROPOUT_RATE = 0.2
    
    # Data
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
```

### 4. Serving Layer (`app_geo.py`)

Flask application serving predictions via REST API.

```mermaid
flowchart TB
    subgraph Auth["Authentication"]
        Login["/login<br/>POST username/password"]
        Session["Flask Session<br/>logged_in flag"]
        Logout["/logout"]
    end
    
    subgraph Endpoints["API Endpoints"]
        GeoPredict["/predict/geo<br/>lat/lon → price"]
        AddrPredict["/predict/address<br/>address → geocode → price"]
        Nearby["/nodes/nearby<br/>Find nodes in radius"]
        Health["/geo/health"]
        Info["/geo/model-info"]
    end
    
    subgraph Processing["Request Processing"]
        Validate["Validate CA Bounds"]
        FindNode["Find Nearest Node"]
        BuildFeat["Build Features"]
        Predict["Model Predict"]
        Classify["Classify Price Level"]
    end
    
    Login --> Session
    Session --> GeoPredict
    Session --> AddrPredict
    
    GeoPredict --> Validate
    AddrPredict --> Validate
    Validate --> FindNode
    FindNode --> BuildFeat
    BuildFeat --> Predict
    Predict --> Classify
```

#### Prediction Flow

```mermaid
sequenceDiagram
    participant Client
    participant Flask as app_geo.py
    participant Geo as geo_utils
    participant Nodes as caiso_nodes
    participant Builder as GeoFeatureBuilder
    participant Model as GeoPricePredictor
    
    Client->>Flask: GET /predict/geo?lat=34.05&lon=-118.24
    Flask->>Geo: is_in_california(lat, lon)
    Geo-->>Flask: True
    
    Flask->>Nodes: find_nearest_node(lat, lon)
    Nodes-->>Flask: (node_id, node_info, distance)
    
    Flask->>Builder: build_all_features(lat, lon, timestamp, ...)
    Builder-->>Flask: {feature_dict}
    
    Flask->>Model: predict_single(features)
    Model->>Model: scale features
    Model->>Model: model.predict()
    Model-->>Flask: price
    
    Flask->>Flask: classify_price_level(price)
    Flask-->>Client: JSON response
```

---

## Data Flow

### Training Data Flow

```mermaid
flowchart LR
    subgraph Sources["Data Sources"]
        S1["CAISO OASIS"]
        S2["NOAA GHCND"]
        S3["CartoDB"]
        S4["caiso-price-map.json"]
    end
    
    subgraph Download["Download Phase"]
        D1["download_geo.py"]
        D2["tier2_pipeline.py"]
    end
    
    subgraph Storage["Raw Storage"]
        R1["geo_prices.pkl"]
        R2["weather_data.pkl"]
        R3["power_plants.pkl"]
        R4["caiso_nodes_california.json"]
    end
    
    subgraph Features["Feature Phase"]
        F1["geo_features.py"]
    end
    
    subgraph Training["Training Phase"]
        T1["train_geo.py"]
    end
    
    subgraph Models["Model Artifacts"]
        M1["geo_model.keras"]
        M2["geo_scaler.pkl"]
        M3["geo_features.json"]
    end
    
    S1 --> D1 --> R1
    S2 --> D2 --> R2
    S3 --> D2 --> R3
    S4 --> R4
    
    R1 --> F1
    R2 --> F1
    R3 --> F1
    R4 --> F1
    
    F1 --> T1
    T1 --> M1
    T1 --> M2
    T1 --> M3
```

### Prediction Data Flow

```mermaid
flowchart LR
    subgraph Request["Client Request"]
        C1["lat/lon coordinates"]
        C2["OR address string"]
    end
    
    subgraph Geocode["Geocoding (if address)"]
        G1["Google Maps API"]
    end
    
    subgraph Features["Feature Building"]
        F1["Time features"]
        F2["Weather interpolation"]
        F3["Spatial features"]
        F4["Default lag values"]
    end
    
    subgraph Model["Model Inference"]
        M1["Load from memory"]
        M2["Scale features"]
        M3["NN forward pass"]
    end
    
    subgraph Response["Response"]
        R1["Price prediction"]
        R2["Level classification"]
        R3["Nearest node info"]
    end
    
    C1 --> Features
    C2 --> G1 --> Features
    
    F1 --> Model
    F2 --> Model
    F3 --> Model
    F4 --> Model
    
    M1 --> M2 --> M3 --> Response
```

---

## Deployment Architecture

### Docker Compose Architecture

```mermaid
flowchart TB
    subgraph Host["Host Machine"]
        subgraph Volumes["Shared Volumes"]
            V1["./src"]
            V2["./data"]
            V3["./templates"]
            V4[".env"]
        end
        
        subgraph Containers["Docker Containers"]
            subgraph Downloader["downloader<br/>(docker-compose-downloader.yml)"]
                DL["download_geo.py"]
            end
            
            subgraph Trainer["trainer<br/>(docker-compose.yml)"]
                TR1["geo_features.py"]
                TR2["train_geo.py"]
            end
            
            subgraph App["app<br/>(docker-compose.yml)"]
                AP["app_geo.py<br/>Port 8001"]
            end
        end
    end
    
    V1 --> Downloader
    V1 --> Trainer
    V1 --> App
    
    V2 --> Downloader
    V2 --> Trainer
    V2 --> App
    
    V4 --> App
```

### Cloud Run Architecture

```mermaid
flowchart TB
    subgraph GCP["Google Cloud Platform"]
        subgraph Registry["Artifact Registry"]
            Image["smart-grid-geo:latest"]
        end
        
        subgraph CloudRun["Cloud Run"]
            Service["smart-grid-geo<br/>2GB Memory"]
            Env["Environment:<br/>GOOGLE_MAPS_API_KEY"]
        end
    end
    
    subgraph Local["Local Build"]
        Dockerfile["Dockerfile.cloudrun"]
        Models["data/models/*"]
        Src["src/*"]
    end
    
    Dockerfile --> Image
    Models --> Image
    Src --> Image
    
    Image --> Service
    Env --> Service
    
    Internet["Internet"] --> Service
```

---

## Security

### Authentication Flow

```mermaid
sequenceDiagram
    participant User
    participant Flask
    participant Session
    
    User->>Flask: GET /
    Flask->>Session: Check logged_in
    Session-->>Flask: False
    Flask-->>User: Redirect to /login
    
    User->>Flask: POST /login (user/2026)
    Flask->>Flask: Validate credentials
    Flask->>Session: Set logged_in=True
    Flask-->>User: Redirect to /
    
    User->>Flask: GET /
    Flask->>Session: Check logged_in
    Session-->>Flask: True
    Flask-->>User: Render index.html
    
    User->>Flask: GET /logout
    Flask->>Session: Clear session
    Flask-->>User: Redirect to /login
```

### Environment Variables

| Variable | Purpose | Storage |
|----------|---------|---------|
| `GOOGLE_MAPS_API_KEY` | Google Maps geocoding | `.env` (gitignored) |
| `AUTH_USERNAME` | Login username | `.env` or default |
| `AUTH_PASSWORD` | Login password | `.env` or default |
| `SECRET_KEY` | Flask session encryption | `.env` or default |

---

## File Structure

```
tier3_poc/
├── src/
│   ├── app_geo.py           # Flask API entry point
│   ├── train_geo.py         # Model training
│   ├── geo_features.py      # Feature engineering
│   ├── download_geo.py      # CAISO data fetcher
│   ├── caiso_nodes.py       # Node management
│   ├── geo_utils.py         # Geospatial utilities
│   ├── tier2_pipeline.py    # NOAA/CartoDB fetchers
│   └── stations.json        # NOAA station list
│
├── data/
│   ├── downloads/           # Raw data
│   │   ├── geo_prices.pkl
│   │   ├── power_plants.pkl
│   │   └── download_geo.log
│   ├── models/              # Trained artifacts
│   │   ├── geo_model.keras
│   │   ├── geo_scaler.pkl
│   │   ├── geo_features.json
│   │   └── geo_metadata.json
│   ├── training/            # Training logs
│   │   └── train_geo.log
│   └── prediction/          # API logs
│       └── app_geo.log
│
├── templates/
│   ├── index.html           # Main UI
│   └── login.html           # Login page
│
├── docker-compose.yml               # Trainer + App
├── docker-compose-downloader.yml    # Downloader
├── Dockerfile                       # Base image
├── Dockerfile.cloudrun              # Production image
└── deploy-to-cloudrun.sh            # GCP deploy script
```

---

## Performance Considerations

### Training Performance

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Training Time | 5-15 min | Depends on data size |
| Memory Usage | 2-4 GB | TensorFlow + pandas |
| Disk Usage | 100-500 MB | Model + training data |

### Inference Performance

| Metric | Typical Value | Notes |
|--------|---------------|-------|
| Cold Start | 3-5 sec | Model loading |
| Inference | 10-50 ms | Per prediction |
| Memory | 500 MB - 1 GB | Model in memory |

### Scaling Recommendations

- **Cloud Run**: Set min instances = 1 to avoid cold starts
- **Memory**: 2 GB minimum for model + feature building
- **Concurrency**: 80 requests/instance (default is fine)

---

## Monitoring

### Log Locations

| Log | Path | Purpose |
|-----|------|---------|
| Download | `data/downloads/download_geo.log` | CAISO fetch progress |
| Features | `data/downloads/geo_features.log` | Feature building |
| Training | `data/training/train_geo.log` | Model training |
| API | `data/prediction/app_geo.log` | Request/response logging |

### Health Checks

```bash
# Check model loaded
curl http://localhost:8001/geo/health

# Check model details
curl http://localhost:8001/geo/model-info
```

---

## Extensibility

### Adding New Features

1. Add feature calculation in `geo_features.py` → `GeoFeatureBuilder`
2. Update `TRAINING_FEATURES` list in `train_geo.py`
3. Retrain model
4. Update `build_prediction_features()` in `app_geo.py`

### Adding New Data Sources

1. Create fetcher in `tier2_pipeline.py`
2. Add to download pipeline in `download_geo.py`
3. Integrate in `geo_features.py`

### Model Architecture Changes

1. Modify `GeoModelConfig` in `train_geo.py`
2. Update `GeoPricePredictor.build_model()`
3. Retrain and deploy
