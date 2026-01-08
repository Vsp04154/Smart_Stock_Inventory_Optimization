import streamlit as st
import mysql.connector
import pandas as pd

# ---------------- DATABASE CONFIG ----------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Vsp@12345",
    "database": "smart_stock"
}

# ---------------- DB CONNECTION ----------------
def get_connection():
    return mysql.connector.connect(**DB_CONFIG)

# ---------------- FETCH LOW STOCK ----------------
def fetch_low_stock():
    conn = get_connection()
    query = """
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
    ORDER BY shortage DESC
    LIMIT 200;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# ---------------- FETCH REORDER ALERTS ----------------
def fetch_reorder_alerts():
    conn = get_connection()
    query = """
    SELECT
        it.store_id,
        it.product_id,
        MAX(it.inventory_level) AS current_stock,
        ROUND((AVG(it.units_sold) * 7) + 50, 2) AS reorder_point,
        ROUND(((AVG(it.units_sold) * 7) + 50) - MAX(it.inventory_level), 2) AS urgency
    FROM inventory_transactions it
    WHERE it.date >= CURDATE() - INTERVAL 14 DAY
    GROUP BY it.store_id, it.product_id
    HAVING current_stock < reorder_point
    ORDER BY urgency DESC;
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def fetch_top_suppliers(product_id, top_n=3):
    # Get product category
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT category FROM products WHERE product_id=%s", (product_id,))
    result = cursor.fetchone()
    if result:
        category = result["category"]
    else:
        category = None

    # Try to pick suppliers based on product category
    if category:
        cursor.execute("""
            SELECT s.supplier_id, s.supplier_name,
                   (s.rating*0.5 + s.discount_percent*0.3 - s.lead_time_days*0.2) AS supplier_score
            FROM suppliers s
            JOIN product_suppliers ps ON s.supplier_id = ps.supplier_id
            JOIN products p ON ps.product_id = p.product_id
            WHERE p.category = %s
            ORDER BY supplier_score DESC
            LIMIT %s
        """, (category, top_n))
        suppliers = cursor.fetchall()
    else:
        suppliers = []

    # Fallback: top suppliers overall if none linked to category
    if not suppliers:
        cursor.execute("""
            SELECT supplier_id, supplier_name,
                   (rating*0.5 + discount_percent*0.3 - lead_time_days*0.2) AS supplier_score
            FROM suppliers
            ORDER BY supplier_score DESC
            LIMIT %s
        """, (top_n,))
        suppliers = cursor.fetchall()

    conn.close()
    return suppliers


# ---------------- PLACE REORDER ----------------
def place_reorder(product_id, quantity):
    top_suppliers = fetch_top_suppliers(product_id)
    if not top_suppliers:
        st.error("❌ No suppliers found!")
        return

    supplier = top_suppliers[0]  # best supplier

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO purchase_orders
        (product_id, supplier_id, order_quantity, order_date, status)
        VALUES (%s, %s, %s, CURDATE(), 'ORDERED')
    """, (product_id, supplier["supplier_id"], int(quantity)))
    conn.commit()
    cursor.close()
    conn.close()

    st.success(f"✅ Reorder placed for {product_id} | Qty: {int(quantity)}")
    st.info(f"Top supplier selected: {supplier['supplier_name']} (Score: {supplier['supplier_score']:.2f})")

    # Show top 3 suppliers
    st.subheader("Supplier Suggestions")
    df_suppliers = pd.DataFrame(top_suppliers)
    st.table(df_suppliers[["supplier_name", "supplier_score"]])

# ---------------- STREAMLIT UI ----------------
st.set_page_config(page_title="Smart Stock", layout="wide")
st.title("📦 Smart Stock – Intelligent Inventory System")

tab1, tab2 = st.tabs(["🚨 Low Stock Alerts", "🔁 Reorder Alerts"])

# ---------------- LOW STOCK TAB ----------------
with tab1:
    st.subheader("Low Stock Alerts")
    df_low = fetch_low_stock()
    
    if not df_low.empty:
        st.markdown("### 🛒 Reorder Products")
        
        # Table header
        cols = st.columns([1,1,1,1,1,1,1,1])
        headers = ["Date", "Store", "Product", "Category", "Inventory", "Demand Forecast", "Shortage", "Action"]
        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")
        st.markdown("---")

        # Table rows with reorder button
        for idx, row in df_low.iterrows():
            cols = st.columns([1,1,1,1,1,1,1,1])
            cols[0].write(row["date"])
            cols[1].write(row["store_id"])
            cols[2].write(row["product_id"])
            cols[3].write(row["category"])
            cols[4].write(row["inventory_level"])
            cols[5].write(row["demand_forecast"])
            cols[6].write(int(row["shortage"]))

            if cols[7].button("Reorder", key=f"low_{idx}"):
                place_reorder(row["product_id"], int(row["shortage"]))
    else:
        st.info("All products are sufficiently stocked.")


# ---------------- REORDER ALERTS TAB ----------------
# ---------------- REORDER ALERTS TAB ----------------
with tab2:
    st.subheader("Reorder Alerts (Urgency Table)")
    df_reorder = fetch_reorder_alerts()
    
    if not df_reorder.empty:
        # Table header
        cols = st.columns([1,1,1,1,1,1,1])
        headers = ["Store", "Product", "Current Stock", "Reorder Point", "Urgency", "Category", "Action"]
        for col, header in zip(cols, headers):
            col.markdown(f"**{header}**")
        st.markdown("---")
        
        for idx, row in df_reorder.iterrows():
            # Get category for supplier selection
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT category FROM products WHERE product_id=%s", (row['product_id'],))
            category = cursor.fetchone()[0] if cursor.rowcount > 0 else "Unknown"
            conn.close()
            
            # Display row
            cols = st.columns([1,1,1,1,1,1,1])
            cols[0].write(row['store_id'])
            cols[1].write(row['product_id'])
            cols[2].write(row['current_stock'])
            cols[3].write(row['reorder_point'])
            cols[4].write(int(row['urgency']))
            cols[5].write(category)
            
            # Reorder button
            if cols[6].button("Reorder", key=f"reorder_{idx}"):
                quantity = max(1, int(row['reorder_point'] - row['current_stock']))
                place_reorder(row['product_id'], quantity)
    else:
        st.info("No products require reordering right now.")
