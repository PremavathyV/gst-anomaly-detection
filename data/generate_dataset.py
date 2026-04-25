# generate_dataset.py — Tamil Nadu GST Synthetic Dataset
# Includes: electricity, freight, employees, water consumption,
# GSTR-1 vs GSTR-3B mismatch, entity resolution fields, audit outcomes

import pandas as pd
import numpy as np

np.random.seed(42)
N = 200

industries = [
    {"nic_code": "1311", "industry": "Textile Weaving",        "energy": 180, "freight": 0.8,  "emp": 0.012, "water": 0.5},
    {"nic_code": "2410", "industry": "Steel/Foundry",          "energy": 350, "freight": 1.2,  "emp": 0.008, "water": 1.2},
    {"nic_code": "1010", "industry": "Food Processing",        "energy": 120, "freight": 1.5,  "emp": 0.015, "water": 2.0},
    {"nic_code": "2910", "industry": "Auto Components",        "energy": 200, "freight": 1.0,  "emp": 0.010, "water": 0.4},
    {"nic_code": "0810", "industry": "Granite Processing",     "energy": 280, "freight": 2.0,  "emp": 0.006, "water": 0.8},
    {"nic_code": "6201", "industry": "Software/IT Services",   "energy": 30,  "freight": 0.05, "emp": 0.025, "water": 0.1},
    {"nic_code": "4610", "industry": "Wholesale Trade",        "energy": 50,  "freight": 2.5,  "emp": 0.005, "water": 0.1},
    {"nic_code": "1701", "industry": "Paper/Packaging",        "energy": 160, "freight": 1.1,  "emp": 0.009, "water": 1.5},
    {"nic_code": "2011", "industry": "Chemical Manufacturing", "energy": 300, "freight": 0.9,  "emp": 0.007, "water": 3.0},
    {"nic_code": "4520", "industry": "Construction",           "energy": 80,  "freight": 1.8,  "emp": 0.020, "water": 0.6},
]

districts = ["Chennai", "Coimbatore", "Madurai", "Salem", "Erode",
             "Tiruppur", "Trichy", "Vellore", "Thoothukudi", "Namakkal"]

rows = []
for i in range(1, N + 1):
    ind  = industries[np.random.randint(0, len(industries))]
    dist = districts[np.random.randint(0, len(districts))]

    true_turnover = np.random.uniform(10, 500)   # ₹ lakhs
    is_evader     = np.random.random() < 0.15

    if is_evader:
        declared_turnover = true_turnover * np.random.uniform(0.30, 0.70)
    else:
        declared_turnover = true_turnover * np.random.uniform(0.95, 1.05)

    # ── Operational signals (based on TRUE turnover) ──
    electricity   = true_turnover * ind["energy"]  * np.random.uniform(0.85, 1.15)
    freight       = true_turnover * ind["freight"] * np.random.uniform(0.85, 1.15)
    employees     = max(1, int(true_turnover * ind["emp"] * np.random.uniform(0.85, 1.15)))
    water_usage   = true_turnover * ind["water"]   * np.random.uniform(0.85, 1.15)  # kilolitres

    # ── GSTR-1 vs GSTR-3B mismatch ──
    # GSTR-1 = outward supply declared; GSTR-3B = tax actually paid
    # Evaders often show a gap between the two
    gstr1_turnover = declared_turnover * 100000   # ₹
    if is_evader:
        # Evader under-reports in GSTR-3B vs GSTR-1 (or vice versa)
        gstr3b_turnover = gstr1_turnover * np.random.uniform(0.60, 0.90)
    else:
        gstr3b_turnover = gstr1_turnover * np.random.uniform(0.97, 1.03)

    gstr_mismatch_pct = round(abs(gstr1_turnover - gstr3b_turnover) / (gstr1_turnover + 1e-9) * 100, 2)

    # ── E-way bill ──
    eway_bill_count = max(1, int(freight * np.random.uniform(0.8, 1.2)))
    eway_bill_value = round(freight * declared_turnover * 100000 * np.random.uniform(0.9, 1.1) / 100, 0)

    # ── Entity resolution fields ──
    gstin    = f"33{str(i).zfill(13)}"
    pan      = f"ABCDE{str(i).zfill(4)}F"
    tangedco = f"TN-{dist[:3].upper()}-{str(np.random.randint(10000,99999))}"
    epfo_id  = f"TN/{str(np.random.randint(1000,9999))}/EPF"
    metro_id = f"MW-{dist[:3].upper()}-{str(np.random.randint(1000,9999))}"  # Metrowater ID

    audited = np.random.random() < 0.30
    audit_outcome = ("EVADER" if is_evader else "CLEAN") if audited else "NOT_AUDITED"

    rows.append({
        "gstin":              gstin,
        "pan":                pan,
        "tangedco_account":   tangedco,
        "epfo_id":            epfo_id,
        "metrowater_id":      metro_id,
        "business_name":      f"{dist} {ind['industry']} Unit {i:03d}",
        "district":           dist,
        "nic_code":           ind["nic_code"],
        "industry_type":      ind["industry"],
        "declared_turnover":  round(declared_turnover * 100000, 0),
        "electricity_units":  round(electricity, 0),
        "employee_count":     employees,
        "freight_movement":   round(freight, 2),
        "water_consumption":  round(water_usage, 2),       # kilolitres/year
        "gstr1_turnover":     round(gstr1_turnover, 0),
        "gstr3b_turnover":    round(gstr3b_turnover, 0),
        "gstr_mismatch_pct":  gstr_mismatch_pct,           # % gap between GSTR-1 and GSTR-3B
        "eway_bill_count":    eway_bill_count,
        "eway_bill_value":    round(eway_bill_value, 0),
        "audit_outcome":      audit_outcome,
        "true_evader":        is_evader,
    })

df = pd.DataFrame(rows)
df.to_csv("data/tamilnadu_gst_data.csv", index=False)
print(f"✅ Generated {len(df)} records → data/tamilnadu_gst_data.csv")
print(f"   Evaders     : {df['true_evader'].sum()} ({df['true_evader'].mean()*100:.1f}%)")
print(f"   Audited     : {(df['audit_outcome'] != 'NOT_AUDITED').sum()}")
print(f"   New columns : water_consumption, gstr_mismatch_pct, eway_bill_count, eway_bill_value, metrowater_id")
