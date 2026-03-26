# 📦 Supply Chain Resilience Modeling using Demand Forecasting (Walmart Dataset)

This project builds an **end-to-end machine learning system** that not only forecasts demand but also translates predictions into **actionable supply chain decisions**.

It integrates:
- 📊 Demand Forecasting (LightGBM)
- 🧠 Explainable AI (SHAP)
- 📦 Inventory Optimization (Reorder Point, Safety Stock)
- ⚖️ Decision Support (AHP-inspired approach)
- ⚡ FastAPI Backend
- 🎛️ Streamlit Interactive Dashboard

---

## 🎯 Objective

To develop a system that:
- Predicts future demand accurately  
- Explains key drivers behind predictions  
- Converts predictions into inventory decisions  
- Improves **supply chain efficiency and resilience**

---

## 🚀 Key Features

- ✅ Machine Learning-based demand forecasting (LightGBM)  
- ✅ SHAP-based explainability for model transparency  
- ✅ Inventory optimization (safety stock, reorder point)  
- ✅ Demand variability simulation for resilience analysis  
- ✅ AHP-inspired multi-criteria decision support  
- ✅ FastAPI for real-time prediction API  
- ✅ Streamlit dashboard for interactive usage  

---

## 🧠 Methodology

### 1. Demand Forecasting
- Used LightGBM to predict weekly sales  
- Feature engineering:
  - Lag features  
  - Rolling averages  
  - Date-based features  

---

### 2. Explainable AI (SHAP)
- Identifies key factors influencing predictions  
- Helps interpret model behavior  

---

### 3. Inventory Optimization

- **Reorder Point**  
  = Average Demand × Lead Time  

- **Safety Stock**  
  = Z × Std Dev × √Lead Time  

- **Inventory Level**  
  = Reorder Point + Safety Stock  

---

### 4. Supply Chain Resilience
- Simulated demand variability  
- Analyzed impact of uncertainty on inventory decisions  

---

### 5. Decision Support (AHP - Simplified)
- Used weighted scoring to rank alternatives  
- Criteria:
  - Demand  
  - Variability  
  - Profitability  
