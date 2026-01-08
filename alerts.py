import mysql.connector

# -----------------------------
# 1. Database connection
# -----------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Vsp@12345",   # 🔴 change this
    database="smart_stock"
)

cursor = conn.cursor(dictionary=True)

# -----------------------------
# 2. LOW STOCK ALERTS
# -----------------------------
low_stock_query = """
SELECT 
    it.date,
    it.store_id,
    it.product_id,
    p.category,
    it.inventory_level,
    it.demand_forecast,
    (it.demand_forecast - it.inventory_level) AS shortage
FROM inventory_transactions it
JOIN products p ON it.product_id = p.product_id
WHERE it.inventory_level < it.demand_forecast
ORDER BY shortage DESC;
"""

cursor.execute(low_stock_query)
low_stock_alerts = cursor.fetchall()

print("\n🚨 LOW STOCK ALERTS")
print("-" * 60)

if not low_stock_alerts:
    print("✅ No low-stock issues found.")
else:
    for row in low_stock_alerts[:10]:  # show top 10
        print(
            f"Date: {row['date']} | "
            f"Store: {row['store_id']} | "
            f"Product: {row['product_id']} ({row['category']}) | "
            f"Stock: {row['inventory_level']} | "
            f"Forecast: {row['demand_forecast']} | "
            f"Shortage: {row['shortage']}"
        )

# -----------------------------
# 3. REORDER ALERTS (Lead time = 7 days)
# -----------------------------
reorder_query = """
SELECT
    store_id,
    product_id,
    ROUND(AVG(units_sold), 2) AS avg_daily_sales,
    MAX(inventory_level) AS current_stock,
    ROUND(AVG(units_sold) * 7, 2) AS reorder_point
FROM inventory_transactions
GROUP BY store_id, product_id
HAVING current_stock < reorder_point;
"""

cursor.execute(reorder_query)
reorder_alerts = cursor.fetchall()

print("\n🔁 REORDER ALERTS")
print("-" * 60)

if not reorder_alerts:
    print("✅ No reorders needed today.")
else:
    for row in reorder_alerts[:10]:
        print(
            f"Store: {row['store_id']} | "
            f"Product: {row['product_id']} | "
            f"Current Stock: {row['current_stock']} | "
            f"Reorder Point: {row['reorder_point']} | "
            f"Avg Daily Sales: {row['avg_daily_sales']}"
        )

# -----------------------------
# 4. Close connection
# -----------------------------
cursor.close()
conn.close()

print("\n✅ Alert fetch completed successfully.")
