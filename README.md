# GST Revenue Anomaly Detection — Tamil Nadu

A multi-source anomaly detection system that identifies businesses likely
under-reporting their GST turnover by cross-validating declared figures
against real-world operational signals.

---

## What This Project Does

Tamil Nadu has ~15 lakh GST-registered dealers. Tax evasion through
turnover under-declaration is a known problem — conservative estimates
suggest 10–15% of potential GST revenue is lost annually.

This system compares what a business **declares** as turnover against
signals they **cannot easily falsify**:

| Signal | Source |
|---|---|
| Electricity consumption | TANGEDCO billing data |
| Employee count | EPFO contribution data |
| Freight movement | GST e-way bill data |
| Water consumption | Chennai Metrowater / municipality |
| GSTR-1 vs GSTR-3B mismatch | GST return filings |

If a business uses electricity consistent with ₹25 crore of production
but declares only ₹5 crore turnover — that gap is flagged as an anomaly.

---

## Project Structure

```
gst_anomaly_detection/
├── data/
│   ├── generate_dataset.py       ← generates synthetic Tamil Nadu dataset
│   └── tamilnadu_gst_data.csv    ← generated dataset (200 businesses)
├── main.py                        ← full CLI pipeline
├── app.py                         ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## Setup — Step by Step

### 1. Make sure Python is installed
Open terminal and check:
```bash
python --version
```
You need Python 3.8 or above.

### 2. Open the project folder in VS Code
```
File → Open Folder → select the gst_anomaly_detection folder
```

### 3. Open the VS Code terminal
```
Terminal → New Terminal  (or press Ctrl + `)
```

### 4. Create a virtual environment
```bash
python -m venv venv
```

### 5. Activate the virtual environment

Windows:
```bash
venv\Scripts\activate
```

Mac / Linux:
```bash
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal line.

### 6. Install all dependencies
```bash
pip install -r requirements.txt
```

---

## Generate the Dataset

Run this once to create the synthetic Tamil Nadu GST dataset:
```bash
python data/generate_dataset.py
```

This creates `data/tamilnadu_gst_data.csv` with 200 businesses across
10 districts and 10 industry types, with realistic operational signals
and labelled audit outcomes.

---

## Run the CLI Pipeline

```bash
python main.py
```

This will:
1. Load and clean the dataset
2. Perform entity resolution (GSTN ↔ TANGEDCO ↔ EPFO ↔ Metrowater)
3. Engineer 6 ratio features with industry-stratified Z-scores
4. Handle class imbalance using data-driven contamination per industry
5. Train Isolation Forest per industry group
6. Generate risk scores (0–100) for every business
7. Print interpretable explanations for each flagged business
8. Evaluate against historical audit outcomes (precision / recall)
9. Save results to `data/gst_anomaly_results.csv`
10. Save charts to `data/analysis_charts.png`

---

## Run the Streamlit Web App

```bash
streamlit run app.py
```

Then open your browser at:
```
http://localhost:8501
```

The app works on both desktop and mobile browsers.

---

## App Tabs Explained

| Tab | What it shows |
|---|---|
| 🗺️ Overview | Risk score by district, suspicious count by industry |
| 📋 Audit List | Full table with filters by district, industry, status |
| 📊 Risk Charts | Bar chart of risk scores, scatter of signals vs turnover |
| 🔬 Industry Analysis | Z-score scatter, box plots per industry |
| 📈 Model Evaluation | Precision, recall, confusion matrix vs audit outcomes |
| 📥 Download | Export full results + audit priority list as CSV |

---

## Upload Your Own Data

In the app sidebar, click **Upload CSV** and upload a file with these columns:

| Column | Description |
|---|---|
| gstin | GST Identification Number (e.g. 33XXXXX) |
| pan | PAN number |
| tangedco_account | TANGEDCO billing account ID |
| epfo_id | EPFO establishment ID |
| metrowater_id | Metrowater connection ID |
| business_name | Name of the business |
| district | District in Tamil Nadu |
| nic_code | NIC industry code (e.g. 1311) |
| industry_type | Industry name (e.g. Textile Weaving) |
| declared_turnover | Annual declared GST turnover in ₹ |
| electricity_units | Annual electricity consumption in kWh |
| employee_count | Number of employees |
| freight_movement | Annual freight volume |
| water_consumption | Annual water consumption in kilolitres |
| gstr_mismatch_pct | % gap between GSTR-1 and GSTR-3B |
| eway_bill_count | Number of e-way bills generated |
| eway_bill_value | Total value of e-way bills in ₹ |
| audit_outcome | EVADER / CLEAN / NOT_AUDITED (optional) |

---

## How the Model Works

```
Raw Data
   ↓
Entity Resolution (match GSTN → TANGEDCO → EPFO → Metrowater)
   ↓
Feature Engineering (6 ratio features)
   ↓
Industry Z-score Normalization (per NIC code group)
   ↓
Isolation Forest (trained per industry, class-imbalance aware)
   ↓
Risk Score 0–100 + Interpretable Explanation
   ↓
Prioritised Audit List
```

**Why Z-scores?**
A software company and a steel foundry with the same turnover have very
different electricity consumption. Raw ratios are misleading. Z-scores
measure how far a business deviates from its own industry peers.

**Why Isolation Forest?**
It is an unsupervised algorithm — it does not need labelled fraud data
to work. It isolates anomalies by randomly splitting data. Normal
businesses need many splits to isolate (they cluster together). Suspicious
businesses are isolated quickly with very few splits (they are outliers).

---

## Expected Output (CLI)

```
Precision (audit hit rate) : 86.67%
Recall (evader catch rate) : 100.00%

TOP SUSPICIOUS BUSINESSES:
B001  Thoothukudi Paper/Packaging  ₹8,34,774  risk=100.0
      ⚡ Electricity 3.9σ | 🚚 Freight 3.9σ | 💧 Water 3.8σ | 📋 GSTR mismatch 10.1%

B002  Salem Granite Processing     ₹4,92,458  risk=88.4
      ⚡ Electricity 3.8σ | 🚚 Freight 3.3σ | 💧 Water 3.6σ | 📋 GSTR mismatch 21.6%
```

---

## How to Improve Accuracy

- Add more quarters of data (time-series trend detection)
- Use XGBoost or Random Forest with the labelled audit outcomes
  for supervised learning once enough labelled data is available
- Add bank transaction data as an additional signal
- Tune contamination parameter per district, not just per industry
- Add seasonal adjustment for industries with peak consumption periods

---

## Dependencies

```
pandas
numpy
scikit-learn
matplotlib
streamlit
plotly
```

Install all with:
```bash
pip install -r requirements.txt
```
