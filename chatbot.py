import mysql.connector
import pandas as pd
import openai
import datetime
import json
import re

# -----------------------------
# CONFIG - put your API key here
# -----------------------------
openai.api_key = ""

# -----------------------------
# MySQL connection
# -----------------------------
conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="smart_stock"
)

TABLE_NAME = "inventory_data"

# -----------------------------
# Load full table once (73k+ rows)
# -----------------------------
df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn)
# normalize column names
df.columns = [c.lower().replace(" ", "_") for c in df.columns]
if 'date' in df.columns:
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

# -----------------------------
# Utilities
# -----------------------------
MONTH_NAMES = "|".join(["january","february","march","april","may","june","july","august","september","october","november","december"])

def call_openai_for_spec(question, schema):
    """
    Ask OpenAI to convert question -> structured JSON spec.
    The model must return ONLY JSON.
    """
    prompt = f"""
You are a Python/Pandas query planner. Given a natural language user question about a DataFrame `df` and the schema below,
produce a JSON object (no explanation, only JSON) that describes exactly how to compute the answer.

Schema columns:
{schema}

Rules for JSON output (required keys):
- "filters": list of {{"column": <string>, "op": <string>, "value": <string|number>}} (op one of: ==, !=, >, <, >=, <=, in)
- "date": optional object for date filtering; one of:
    - {{"type":"last_month"}} 
    - {{"type":"last_week"}}
    - {{"type":"month_year", "month":"October", "year":2023}}
    - {{"type":"range", "start":"YYYY-MM-DD", "end":"YYYY-MM-DD"}}
- "derived": optional {{"name":"revenue", "expr":"price * units_sold"}} — a short arithmetic expression using column names and operators + - * /
- "groupby": optional string or list of strings (e.g. "product_id" or ["category","product_id"])
- "aggregation": optional object {{"func":"sum"|"mean"|"count"|"max"|"min", "column":"units_sold"}}
- "top_n": optional integer (1 means top 1)
- "return": one of "single_value", "row", "table", "series" (if omitted, prefer single_value when possible)

Examples (these are examples — DO NOT RETURN these; return JSON only for the user's question):
- Top selling product in Electronics in October 2023:
  {{ "filters":[{{"column":"category","op":"==","value":"Electronics"}}], "date":{{"type":"month_year","month":"October","year":2023}}, "groupby":"product_id", "aggregation":{{"func":"sum","column":"units_sold"}}, "top_n":1, "return":"row" }}

User question:
\"\"\"{question}\"\"\"
Return strictly one JSON object following the rules above.
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system","content":"You are a strict JSON-only generator that outputs a query spec for Pandas."},
            {"role":"user","content":prompt}
        ],
        temperature=0.0,
        max_tokens=600
    )
    text = resp.choices[0].message.content.strip()
    # try to extract JSON substring if model includes backticks or text
    try:
        # find first { and last }
        start = text.find("{")
        end = text.rfind("}") + 1
        json_text = text[start:end]
        spec = json.loads(json_text)
        return spec
    except Exception as e:
        raise ValueError(f"OpenAI did not return valid JSON spec. Raw response: {text}\nError: {e}")

def apply_filters(base_df, filters):
    df_loc = base_df
    for f in filters or []:
        col = f.get("column")
        op = f.get("op")
        val = f.get("value")
        if col not in df_loc.columns:
            # try case-insensitive column match
            matches = [c for c in df_loc.columns if c.lower() == col.lower()]
            if matches:
                col = matches[0]
            else:
                raise KeyError(f"Column '{col}' not found in dataframe.")
        # handle 'in' operator for comma-separated or list values
        if op == "in":
            if isinstance(val, str):
                vals = [v.strip() for v in val.split(",")]
            elif isinstance(val, list):
                vals = val
            else:
                vals = [val]
            df_loc = df_loc[df_loc[col].astype(str).str.lower().isin([str(x).lower() for x in vals])]
        else:
            # coerce types where possible
            series = df_loc[col]
            # attempt numeric conversion
            try:
                if isinstance(val, (int, float)):
                    pass
                else:
                    # if column numeric, cast value
                    if pd.api.types.is_numeric_dtype(series):
                        if isinstance(val, str) and val.isdigit():
                            val = float(val)
                # apply ops
            except:
                pass
            if op == "==":
                df_loc = df_loc[series.astype(str).str.lower() == str(val).lower()] if series.dtype == object else df_loc[series == val]
            elif op == "!=":
                df_loc = df_loc[series.astype(str).str.lower() != str(val).lower()] if series.dtype == object else df_loc[series != val]
            elif op == ">":
                df_loc = df_loc[series > float(val)]
            elif op == "<":
                df_loc = df_loc[series < float(val)]
            elif op == ">=":
                df_loc = df_loc[series >= float(val)]
            elif op == "<=":
                df_loc = df_loc[series <= float(val)]
            else:
                raise ValueError(f"Unsupported operator: {op}")
    return df_loc

def apply_date_filter(base_df, date_spec):
    if date_spec is None:
        return base_df
    if 'date' not in base_df.columns:
        return base_df
    today = datetime.datetime.today()
    t = date_spec.get("type")
    if t == "last_month":
        first_day = (today.replace(day=1) - pd.Timedelta(days=1)).replace(day=1)
        last_day = today.replace(day=1) - pd.Timedelta(days=1)
        return base_df[(base_df['date'] >= first_day) & (base_df['date'] <= last_day)]
    if t == "last_week":
        start = today - pd.Timedelta(days=today.weekday()+7)
        end = start + pd.Timedelta(days=6)
        return base_df[(base_df['date'] >= start) & (base_df['date'] <= end)]
    if t == "month_year":
        month_str = date_spec.get("month")
        year = int(date_spec.get("year"))
        month = datetime.datetime.strptime(month_str, "%B").month
        return base_df[(base_df['date'].dt.month == month) & (base_df['date'].dt.year == year)]
    if t == "range":
        start = pd.to_datetime(date_spec.get("start"))
        end = pd.to_datetime(date_spec.get("end"))
        return base_df[(base_df['date'] >= start) & (base_df['date'] <= end)]
    return base_df

def safe_create_derived(df_loc, derived):
    """
    derived: {"name":"revenue", "expr":"price * units_sold"}
    We'll use df.eval to compute this column safely.
    Only allow letters, digits, underscores, spaces and operators +-*/() and dots.
    """
    if not derived:
        return df_loc
    name = derived.get("name")
    expr = derived.get("expr")
    if not name or not expr:
        return df_loc
    # simple safety check
    if not re.fullmatch(r"[0-9A-Za-z_ \+\-\*\/\.\(\)]+", expr.replace(" ", "")):
        raise ValueError("Derived expression contains disallowed characters.")
    # ensure referenced columns exist
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr)
    for tok in tokens:
        if tok not in df_loc.columns:
            # allow numeric names? else error
            raise KeyError(f"Derived expression references unknown column '{tok}'.")
    # compute
    df_loc = df_loc.copy()
    # use pandas.eval (safer than exec)
    df_loc[name] = df_loc.eval(expr)
    return df_loc

def compute_from_spec(spec):
    """
    Given the validated spec (dict), compute a Pandas result.
    Returns a Python object result (scalar, dict, list, or small dataframe converted to list-of-dicts).
    """
    df_loc = df.copy()

    # date filter
    df_loc = apply_date_filter(df_loc, spec.get("date"))

    # filters
    df_loc = apply_filters(df_loc, spec.get("filters", []))

    # handle empty
    if df_loc.empty:
        return {"type":"empty", "value": "No matching records for the requested filters/date."}

    # derived column
    if spec.get("derived"):
        df_loc = safe_create_derived(df_loc, spec.get("derived"))

    # grouping and aggregation
    groupby = spec.get("groupby")
    agg = spec.get("aggregation")
    top_n = spec.get("top_n", None)
    ret_type = spec.get("return", None)

    # if aggregation + groupby -> series or dataframe
    if groupby and agg:
        # groupby can be string or list
        gb_cols = groupby if isinstance(groupby, list) else [groupby]
        func = agg.get("func")
        col = agg.get("column")
        if col not in df_loc.columns:
            raise KeyError(f"Aggregation column '{col}' not found.")
        grouped = df_loc.groupby(gb_cols)[col].agg(func)
        if grouped.empty:
            return {"type":"empty", "value":"No data after grouping/aggregation."}
        # top_n
        if top_n:
            top = grouped.sort_values(ascending=False).head(int(top_n))
            # convert to list of dicts
            out = []
            for idx, val in top.items():
                if isinstance(idx, tuple):
                    idx_keys = {gb_cols[i]: idx[i] for i in range(len(gb_cols))}
                else:
                    idx_keys = {gb_cols[0]: idx}
                idx_keys["value"] = val
                out.append(idx_keys)
            return {"type":"table", "value": out}
        else:
            # if top_n not requested and single groupby -> return best item
            if isinstance(grouped, pd.Series):
                best_idx = grouped.idxmax()
                best_val = grouped.max()
                return {"type":"row", "value": {"group": best_idx, "value": best_val}}
            else:
                return {"type":"table", "value": grouped.reset_index().to_dict(orient="records")}

    # if aggregation only
    if agg and not groupby:
        func = agg.get("func")
        col = agg.get("column")
        if col not in df_loc.columns:
            raise KeyError(f"Aggregation column '{col}' not found.")
        if func == "sum":
            return {"type":"single", "value": float(df_loc[col].sum())}
        if func == "mean":
            return {"type":"single", "value": float(df_loc[col].mean())}
        if func == "count":
            return {"type":"single", "value": int(df_loc[col].count())}
        if func == "max":
            return {"type":"single", "value": float(df_loc[col].max())}
        if func == "min":
            return {"type":"single", "value": float(df_loc[col].min())}

    # no aggregation/groupby: try to infer intent
    # If top_n requested and grouping by product_id likely wanted top products by units_sold
    if top_n and "units_sold" in df_loc.columns:
        grouped = df_loc.groupby("product_id")["units_sold"].sum().sort_values(ascending=False).head(int(top_n))
        out = []
        for pid, val in grouped.items():
            row = {"product_id": pid, "units_sold": int(val)}
            if "product_name" in df.columns:
                name = df[df['product_id'] == pid]['product_name'].iloc[0]
                row["product_name"] = name
            out.append(row)
        return {"type":"table", "value": out}

    # fallback: if there's a natural-sounding column to summarize, return basic stats for numeric columns
    numeric = df_loc.select_dtypes(include="number").columns.tolist()
    if numeric:
        summary = {col: {"sum": float(df_loc[col].sum()), "mean": float(df_loc[col].mean()), "max": float(df_loc[col].max())} for col in numeric}
        return {"type":"summary", "value": summary}

    # last fallback
    return {"type":"unknown", "value":"Could not infer computation to perform."}

def rephrase_result(result_obj, original_question):
    """
    Send the computed result (converted to a short string) to OpenAI to rephrase into one nice sentence.
    """
    # build short textual result
    if result_obj.get("type") == "empty":
        result_text = result_obj.get("value")
    elif result_obj.get("type") in ("single", "row"):
        result_text = str(result_obj.get("value"))
    elif result_obj.get("type") == "table":
        # show top 5 entries compactly
        rows = result_obj.get("value")
        if not rows:
            result_text = "No results."
        else:
            snippets = []
            for r in rows[:5]:
                snippets.append(", ".join([f"{k}: {v}" for k, v in r.items()]))
            result_text = "; ".join(snippets)
    elif result_obj.get("type") == "summary":
        # convert summary to short string
        s = result_obj.get("value")
        items = []
        for col, stats in s.items():
            items.append(f"{col} sum={round(stats['sum'],2)}, mean={round(stats['mean'],2)}")
        result_text = " | ".join(items[:5])
    else:
        result_text = str(result_obj.get("value"))

    prompt = f"""
You are an expert, friendly AI assistant. Rephrase the following computation result into a single natural, helpful, and impressive sentence.
Keep it concise and mention the user's original question in a small way if appropriate.

Original question: "{original_question}"
Computation result: "{result_text}"
"""
    resp = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role":"system", "content":"You are a helpful AI assistant that writes a single polished sentence."},
            {"role":"user", "content":prompt}
        ],
        temperature=0.2,
        max_tokens=150
    )
    return resp.choices[0].message.content.strip()

# -----------------------------
# Main orchestrator
# -----------------------------
def ask_smart_chatbot(question):
    try:
        schema = ", ".join(df.columns)
        spec = call_openai_for_spec(question, schema)
        # validate basic shape
        if not isinstance(spec, dict):
            return "Failed to get a valid plan from AI."
        # compute result
        computed = compute_from_spec(spec)
        # rephrase for user
        return rephrase_result(computed, question)
    except Exception as e:
        return f"Could not compute answer: {e}"

# -----------------------------
# Interactive loop
# -----------------------------
if __name__ == "__main__":
    print("Smart Stock Chatbot (type 'thank you' to exit)")
    while True:
        user_input = input("\nYou: ")
        if user_input.strip().lower() in ["thank you", "thanks"]:
            print("Chatbot: You're welcome! Goodbye.")
            break
        reply = ask_smart_chatbot(user_input)
        print(f"Chatbot: {reply}")
