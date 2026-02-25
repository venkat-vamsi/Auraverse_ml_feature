📘 AURAVERSE – Panic Attack Detection ML Module

🔹 SECTION 1: COMPLETE SETUP GUIDE (FROM SCRATCH)

This section explains how to set up and run the ML system step-by-step.

✅ Step 0: Install Python

Install Python 3.9 or above

During installation, check:

✔ Add Python to PATH

Verify installation:

python --version
✅ Step 1: Clone / Download Project

Place the project folder anywhere on your system.

Open terminal inside project root:

good doctor ml feature/
✅ Step 2: Install Required Libraries

Run:

pip install -r requirements.txt

If requirements.txt does not exist, install manually:

pip install pandas scikit-learn joblib numpy
✅ Step 3: Dataset Placement

Place dataset files in this structure:

data/
   train/
      Final_Dataframe_P1.csv
      ...
      Final_Dataframe_P10.csv

   test/
      Final_Dataframe_P11.csv
      ...
      Final_Dataframe_P14.csv

✔ train/ → used for model training
✔ test/ → used for final testing

✅ Step 4: Run the ML Pipeline

From project root:

python main.py
✅ Step 5: Output Location

After execution:

data/output/test_panic_with_cause.csv

This file contains:

All detected panic events in test set

Assigned cause of panic








🔹 SECTION 2: COMPLETE PROJECT EXPLANATION (FOR PRESENTATION)
🧠 Project Overview

This ML module is part of the AURAVERSE system.

Its purpose is to:

Detect panic attacks using wearable sensor data.

Identify the environmental cause of each detected panic attack.

Enable real-time alerting and end-of-day analysis.

The system works in two stages.

🔹 Stage 1 – Panic Detection (Isolation Forest)
🎯 Goal:

Detect abnormal physiological states.

📥 Input Features:

Heart Rate

Skin Resistance

Temperature

🧠 Model Used:

Isolation Forest (Anomaly Detection)

Why Isolation Forest?

Panic events are rare.

It detects outliers in physiological patterns.

It does not assume balanced classes.

It is suitable for unsupervised anomaly detection.

Output:

Normal state

Panic state (anomaly)

Only panic rows are passed to Stage 2.

🔹 Stage 2 – Cause Identification (KMeans Clustering)
🎯 Goal:

Identify what triggered the panic.

📥 Input Features:

Lux (Light intensity)

Sound level

Humidity

🧠 Model Used:

KMeans Clustering (K=3)

Why KMeans?

Groups panic events into environmental patterns.

Identifies dominant trigger per cluster.

Cluster Meaning Assignment:

Each cluster is labeled based on highest dominant feature:

High Lux → Light-triggered

High Sound → Sound-triggered

Otherwise → Environmental-triggered

🔹 Complete Data Flow
Wearable Sensor Data
        ↓
Stage 1 (Isolation Forest)
        ↓
Panic Events Identified
        ↓
Stage 2 (KMeans)
        ↓
Cause Assigned
        ↓
Output CSV Generated
🔹 Why File-Based Split?

Files P1–P10 → Training
Files P11–P14 → Testing

This ensures:

No data leakage

Proper evaluation

Subject-level separation

🔹 Real-Time + End-of-Day Concept

In the full AURAVERSE system:

Live sensor data → Stage 1 → If panic → Twilio call alert to parents.

All panic rows stored.

End-of-day analysis → Stage 2 cause identification.

🔹 Why This Architecture Is Strong

✔ Unsupervised learning
✔ Handles imbalanced panic events
✔ Clean separation of detection and cause
✔ Scalable design
✔ Real-world applicable