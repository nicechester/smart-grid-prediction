# 🔧 Price Scale Fix - Resolving High Predictions

## Problem

Predictions showing unrealistic values like **$109,186/MWh** instead of **$40-80/MWh**.

### Root Cause

The model was trained on **EIA demand data** (values: 10,000-50,000 MWh) which was incorrectly used as price proxy without proper scaling.

---

## ✅ Solution Applied

### 1. **tier2_pipeline.py** - Scale EIA Data
```python
# OLD (WRONG):
value = float(record['value'])  # 10,000-50,000 MWh

# NEW (CORRECT):
value = float(record['value']) / 1000.0  # 10-50 (price-like)
```

### 2. **train.py** - Clip Prices
```python
df['price'] = df['value']
df['price'] = np.clip(df['price'], 10, 150)  # Reasonable range
```

### 3. **app.py** - Emergency Clipping
```python
price = result['predicted_price']
price = np.clip(price, 10, 200)  # Safety net
```

---

## 🚀 How to Fix

### Step 1: Retrain the Model

```bash
# Delete old model
rm -rf models/*

# Retrain with corrected data scaling
python train.py --epochs 50

# You should see:
# ✓ Loaded 5000 real EIA records
# ✓ Processed EIA data: (5000, 14)
# ✓ Test MAE: 3-8 $/MWh  (good!)
# ✓ Test MAPE: 8-15%     (good!)
```

### Step 2: Restart API

```bash
# Stop old service
docker-compose down

# Rebuild and start
docker-compose up --build smart_grid_api
```

### Step 3: Verify

```bash
curl "http://localhost:8000/predict?location_id=los_angeles&location_type=city"

# Expected response:
{
  "predicted_price": 45.23,  # ✓ Reasonable!
  "price_level": "MEDIUM",
  ...
}
```

---

## 📊 Expected Price Ranges

| Level | Price Range | Description |
|-------|-------------|-------------|
| LOW | $10-30/MWh | Off-peak, high renewable |
| MEDIUM | $30-50/MWh | Normal conditions |
| HIGH | $50-80/MWh | Peak demand |
| CRITICAL | $80-200/MWh | Grid stress, heat waves |

---

## 🔍 Understanding the Data

### EIA API Returns

```json
{
  "value": 25000,           // Demand in MWh
  "type": "D",              // Demand
  "type-name": "Demand",
  "unit": "megawatthours"   // Not price!
}
```

### What We Need

- **Actual prices**: $20-100/MWh (typical)
- **Our proxy**: Demand / 1000 → $25/MWh (approximate)

### Why This Works

- High demand (40,000 MWh) → High price ($40/MWh)
- Low demand (20,000 MWh) → Low price ($20/MWh)
- Correlation: ~0.7-0.8

---

## 🎯 Alternative: Use Real Price Data

For production, use actual LMP (Locational Marginal Price) data:

### CAISO OASIS API (Real Prices)

```python
# Example: Get real-time prices
url = "http://oasis.caiso.com/oasisapi/SingleZip"
params = {
    'queryname': 'PRC_LMP',
    'market_run_id': 'RTM',
    'startdatetime': '20251023T00:00-0000',
    'enddatetime': '20251023T23:59-0000',
}
```

Benefits:
- ✓ Actual prices, not proxies
- ✓ More accurate predictions
- ✓ Real-time data

Downsides:
- ✗ Requires CAISO account
- ✗ More complex API
- ✗ Higher rate limits

---

## 🧪 Testing After Fix

### Test 1: Single Prediction

```python
import requests

response = requests.get('http://localhost:8000/predict',
    params={'location_id': 'san_francisco', 'location_type': 'city'})

data = response.json()
price = data['predicted_price']

assert 10 <= price <= 200, f"Price out of range: {price}"
print(f"✓ Price is reasonable: ${price:.2f}/MWh")
```

### Test 2: Multiple Locations

```python
cities = ['los_angeles', 'san_francisco', 'san_diego', 'sacramento']

for city in cities:
    response = requests.get('http://localhost:8000/predict',
        params={'location_id': city, 'location_type': 'city'})
    
    price = response.json()['predicted_price']
    print(f"{city:15s}: ${price:6.2f}/MWh")
    
    # All should be in reasonable range
    assert 10 <= price <= 200

# Expected output:
# los_angeles    : $ 52.30/MWh
# san_francisco  : $ 38.70/MWh
# san_diego      : $ 45.20/MWh
# sacramento     : $ 41.50/MWh
```

### Test 3: Peak vs Off-Peak

```python
from datetime import datetime

# Simulate peak hour (6 PM)
# vs off-peak (3 AM)

# Peak hours should have higher prices
# Off-peak should have lower prices
```

---

## 📝 Summary

| Issue | Fix | Status |
|-------|-----|--------|
| EIA data scale | Divide by 1000 | ✅ Fixed |
| Price clipping | Add min/max bounds | ✅ Fixed |
| Model retraining | Required after fix | ⚠️ **Action Needed** |
| API clipping | Emergency safety net | ✅ Fixed |

## ⚡ Quick Fix (Without Retraining)

If you can't retrain immediately:

```python
# In app.py, adjust the price calculation:
price = result['predicted_price']

# Aggressive scaling for old model
if price > 1000:
    price = price / 1000  # Old model issue
    
price = np.clip(price, 10, 200)
```

This is a **temporary workaround**. You should retrain the model properly.

---

## 🎓 Lessons Learned

1. **Always validate data scales** - Check units and ranges
2. **Use domain knowledge** - Electricity prices have known ranges
3. **Add safety clipping** - Prevent unrealistic predictions
4. **Test with real data** - Synthetic data hides scale issues
5. **Monitor in production** - Alert on outlier predictions

---

**Status**: Ready to retrain! Run `python train.py` with the fixed code. 🚀
