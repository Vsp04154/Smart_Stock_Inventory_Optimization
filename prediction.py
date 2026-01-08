import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ===========================
# 1. Page Config
# ===========================
st.set_page_config(page_title="Smart Stock Prediction", layout="wide")
st.title("📊 Smart Stock Prediction & Alerts Dashboard")

# ===========================
# 2. Connect to MySQL Database
# ===========================
DATABASE_URI = "mysql+mysqlconnector://root:Vsp%4012345@localhost/smart_stock"

@st.cache_data
def load_data():
    engine = create_engine(DATABASE_URI)
    query = """
    SELECT product_id, category, inventory_level, units_sold,
           units_ordered, demand_forecast, price, discount
    FROM inventory_data
    ORDER BY date ASC
    LIMIT 5000
    """
    return pd.read_sql(query, engine)

df = load_data()

st.subheader("📌 Sample Data Preview")
st.dataframe(df.head(10))

# ===========================
# 3. Train Random Forest Model
# ===========================
features = ['units_sold', 'units_ordered', 'demand_forecast', 'price', 'discount']
X = df[features]
y = df['units_sold']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

df['Predicted_Units_Sold'] = model.predict(X)

# ===========================
# 4. Compute Inventory Forecast
# ===========================
df['Predicted_Inventory'] = df['inventory_level'] + df['units_ordered'] - df['Predicted_Units_Sold']

# ===========================
# 5. Actual vs Predicted Bar Graph
# ===========================
st.subheader("📈 Actual vs Predicted Units Sold (First 200 Records)")

fig, ax = plt.subplots(figsize=(14,6))
indices = range(min(200, len(df)))
width = 0.40

ax.bar([i - width/2 for i in indices], df['units_sold'][:200], width=width, label="Actual", color='#4aa3ff')
ax.bar([i + width/2 for i in indices], df['Predicted_Units_Sold'][:200], width=width, label="Predicted", color='#ffae42')

ax.set_title("Units Sold Comparison", fontsize=16, fontweight='bold')
ax.set_xlabel("Record Index")
ax.set_ylabel("Units Sold")
ax.legend()
ax.grid(axis='y', alpha=0.4)

st.pyplot(fig)

# ===========================
# 6. Display Stock Alerts (UPDATED AS PER REQUIREMENT)
# ===========================
st.subheader("📢 Predicted Stock Alerts (Based on Threshold 500 Units)")

THRESHOLD = 500

alert_df = df.head(20).copy()   # Only first 20 rows

def new_stock_status(units):
    if units == 0:
        return "Empty Stock"
    elif units < 200:
        return "Low Stock"
    elif units > THRESHOLD:
        return "Overstock"
    else:
        return "Normal"

alert_df['Stock_Status'] = alert_df['inventory_level'].apply(new_stock_status)

status_colors = {
    'Empty Stock': '#ff4d4d',      # Red
    'Low Stock': '#ffa500',        # Orange
    'Overstock': '#90EE90',        # Light Green
    'Normal': '#D3D3D3'            # Grey
}

status_icons = {
    'Empty Stock': '❌',
    'Low Stock': '⚠️',
    'Overstock': '📦',
    'Normal': '✔️'
}

for _, row in alert_df.iterrows():

    # Reorder logic based on status & prediction
    if row['Stock_Status'] in ["Empty Stock", "Low Stock"]:
        reorder_msg = "<b>🚨 Reorder Required Immediately!</b>"
    elif row['Stock_Status'] == "Normal" and row['Predicted_Units_Sold'] > row['inventory_level']:
        reorder_msg = "<b>🔔 Prediction Higher — Reorder Suggested</b>"
    else:
        reorder_msg = "<b>Stable — No Reorder</b>"

    st.markdown(
        f"<div style='background-color:{status_colors[row['Stock_Status']]}; "
        f"color:black; padding:12px; border-radius:5px; margin-bottom:10px;'>"
        f"{status_icons[row['Stock_Status']]} <b>Product {row['product_id']} ({row['category']})</b><br><br>"
        f"🟢 <b>Actual Units:</b> {int(row['inventory_level'])}<br>"
        f"📌 <b>Stock Status:</b> {row['Stock_Status']}<br>"
        f"🔮 <b>Predicted Units Sold:</b> {int(row['Predicted_Units_Sold'])}<br><br>"
        f"{reorder_msg}"
        f"</div>", unsafe_allow_html=True
    )
