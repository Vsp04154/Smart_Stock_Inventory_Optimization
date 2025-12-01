# smart_inventory_full_report.py

import pandas as pd
from sqlalchemy import create_engine, text
from urllib.parse import quote_plus
import matplotlib.pyplot as plt

# -----------------------------
# CONFIGURATION
# -----------------------------
MYSQL_USER = "root"
MYSQL_PASSWORD = "Vsp@12345"  # Replace with your MySQL password
MYSQL_HOST = "localhost"
MYSQL_DB = "retail_inventory"

CSV_FILE = "C:/ProgramData/MySQL/MySQL Server 8.0/Uploads/retail_store_inventory.csv"

# -----------------------------
# ENCODE PASSWORD
# -----------------------------
password_encoded = quote_plus(MYSQL_PASSWORD)

# -----------------------------
# CREATE SQLALCHEMY ENGINE
# -----------------------------
engine = create_engine(f"mysql+pymysql://{MYSQL_USER}:{password_encoded}@{MYSQL_HOST}/{MYSQL_DB}")

# -----------------------------
# TEST CONNECTION
# -----------------------------
with engine.connect() as conn:
    result = conn.execute(text("SELECT NOW()"))
    print("="*60)
    print("SMART INVENTORY REPORT")
    print("Date:", pd.Timestamp.today().strftime("%Y-%m-%d"))
    print("MySQL Server Time:", result.fetchone()[0])
    print("="*60)

# -----------------------------
# LOAD DATA INTO PANDAS
# -----------------------------
df = pd.read_sql("SELECT * FROM retail_store_inventory", engine)

# -----------------------------
# CLEAN DATA
# -----------------------------
text_columns = ['seasonality','category','region','weather_condition']
for col in text_columns:
    df[col] = df[col].str.strip()

# -----------------------------
# ANALYTICS
# -----------------------------

# 1. Total inventory per store
inventory_per_store = df.groupby("store_id")["inventory_level"].sum()
inventory_per_store_formatted = inventory_per_store.apply(lambda x: f"{x:,}")
print("\n--- Total Inventory per Store ---")
print(inventory_per_store_formatted)

# 2. Total units sold per category
units_sold_category = df.groupby("category")["units_sold"].sum()
units_sold_category_formatted = units_sold_category.apply(lambda x: f"{x:,}")
print("\n--- Total Units Sold per Category ---")
print(units_sold_category_formatted)

# 3. Average product price
avg_price = df["price"].mean()
print("\nAverage Product Price: ₹{:.2f}".format(avg_price))

# 4. Low inventory alert
low_inventory = df[df["inventory_level"] < 50][["store_id","product_id","inventory_level"]]
if not low_inventory.empty:
    print("\n⚠️ Low Inventory Alert (<50 units)")
    print(low_inventory)
else:
    print("\n✅ All products have sufficient inventory.")

# 5. Top 5 best-selling products
top_selling = df.groupby("product_id")["units_sold"].sum().sort_values(ascending=False).head(5)
print("\n--- Top 5 Best-Selling Products ---")
print(top_selling.apply(lambda x: f"{x:,} units"))

# 6. Store summary table (inventory + units sold)
store_summary = pd.DataFrame({
    "Total Inventory": inventory_per_store,
    "Units Sold": df.groupby("store_id")["units_sold"].sum()
})
store_summary = store_summary.apply(lambda col: col.map(lambda x: f"{x:,}"))
print("\n--- Store Summary ---")
print(store_summary)

# -----------------------------
# 7. STOCK LEVEL ANALYSIS
# -----------------------------
def stock_level(row):
    if row < 50:
        return "Low"
    elif row < 200:
        return "Medium"
    else:
        return "High"

df['Stock Level'] = df['inventory_level'].apply(stock_level)

# Overall stock level summary
stock_summary = df.groupby('Stock Level')['product_id'].count()
print("\n--- Stock Level Summary ---")
print(stock_summary)

# Stock level per store
store_stock_summary = df.groupby(['store_id','Stock Level'])['product_id'].count().unstack(fill_value=0)
print("\n--- Stock Level per Store ---")
print(store_stock_summary)

# -----------------------------
# VISUALIZATION
# -----------------------------

# Inventory per store
plt.figure(figsize=(8,5))
inventory_per_store.plot(kind='bar', color='skyblue', title='Total Inventory per Store')
plt.ylabel('Inventory Level')
plt.xlabel('Store ID')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Units sold per category
plt.figure(figsize=(8,5))
units_sold_category.plot(kind='bar', color='orange', title='Units Sold per Category')
plt.ylabel('Units Sold')
plt.xlabel('Category')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# Stock level summary chart
plt.figure(figsize=(6,4))
stock_summary.plot(kind='bar', color=['red','orange','green'], title='Products by Stock Level')
plt.ylabel('Number of Products')
plt.xlabel('Stock Level')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Stock level per store chart
store_stock_summary.plot(kind='bar', stacked=True, figsize=(8,5),
                         color=['red','orange','green'], title='Stock Level per Store')
plt.ylabel('Number of Products')
plt.xlabel('Store ID')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
