"""
Manufacturing Quality Dashboard (Dash + Plotly)

Reads the outputs of etl.py + train_model.py and renders three panels aimed at a
non-technical stakeholder (a quality manager, not a data scientist):

  1. Defect rate over time -- "is quality getting better or worse?"
  2. Top contributing sensors -- "which sensors should engineers look at first?"
  3. Flagged units table -- "which specific units should we pull for inspection today?"

Run:
    python3 app.py
Then open http://127.0.0.1:8050 in your browser.
"""

import pandas as pd
from pathlib import Path
import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go

OUTPUTS = Path(__file__).parent / "outputs"
scores = pd.read_csv(OUTPUTS / "risk_scores.csv", parse_dates=["Timestamp"])
importance = pd.read_csv(OUTPUTS / "feature_importance.csv")

INSPECTION_BUDGET = int(scores["flagged_for_inspection"].sum())
TOTAL_DEFECTS = int(scores["actual_defect"].sum())
CAUGHT = int(scores.loc[scores["flagged_for_inspection"], "actual_defect"].sum())

app = dash.Dash(__name__)
app.title = "Quality Data Lab Dashboard"

# --- Panel 1: defect rate over time, binned into daily batches ---
scores_binned = scores.set_index("Timestamp").resample("1D")["actual_defect"].mean().reset_index()
fig_trend = go.Figure()
fig_trend.add_trace(go.Scatter(
    x=scores_binned["Timestamp"], y=scores_binned["actual_defect"] * 100,
    mode="lines+markers", line=dict(color="#E31937", width=2),
    name="Defect rate (%)",
))
fig_trend.update_layout(
    title="Defect Rate Over Time (test period)",
    yaxis_title="Defect rate (%)", xaxis_title="Date",
    template="plotly_white", margin=dict(l=40, r=20, t=50, b=40),
)

# --- Panel 2: top contributing sensors ---
fig_importance = go.Figure(go.Bar(
    x=importance["importance"], y=importance["sensor"],
    orientation="h", marker_color="#1a1a1a",
))
fig_importance.update_layout(
    title="Top 15 Sensors Driving Defect Predictions",
    xaxis_title="Model importance", yaxis=dict(autorange="reversed"),
    template="plotly_white", margin=dict(l=100, r=20, t=50, b=40),
)

# --- Panel 3: flagged units table ---
flagged_table = scores[scores["flagged_for_inspection"]].copy()
flagged_table = flagged_table[["Timestamp", "risk_score", "rank", "actual_defect"]]
flagged_table["risk_score"] = flagged_table["risk_score"].round(3)
flagged_table["actual_defect"] = flagged_table["actual_defect"].map({1: "Defect", 0: "Pass"})
flagged_table["Timestamp"] = flagged_table["Timestamp"].dt.strftime("%Y-%m-%d %H:%M")

app.layout = html.Div(style={"fontFamily": "Arial, sans-serif", "padding": "24px", "backgroundColor": "#fafafa"}, children=[
    html.H1("Manufacturing Quality Dashboard", style={"marginBottom": "4px"}),
    html.P("Defect detection and inspection alert prioritization for the production line.",
           style={"color": "#555", "marginTop": "0"}),

    html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "24px"}, children=[
        html.Div(style={"flex": 1, "backgroundColor": "white", "padding": "16px", "borderRadius": "8px",
                         "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"}, children=[
            html.H4("Inspection Budget", style={"margin": "0 0 4px 0", "color": "#555"}),
            html.H2(f"{INSPECTION_BUDGET} units", style={"margin": 0}),
        ]),
        html.Div(style={"flex": 1, "backgroundColor": "white", "padding": "16px", "borderRadius": "8px",
                         "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"}, children=[
            html.H4("Defects Caught", style={"margin": "0 0 4px 0", "color": "#555"}),
            html.H2(f"{CAUGHT} of {TOTAL_DEFECTS}", style={"margin": 0}),
        ]),
        html.Div(style={"flex": 1, "backgroundColor": "white", "padding": "16px", "borderRadius": "8px",
                         "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"}, children=[
            html.H4("Recall at Budget", style={"margin": "0 0 4px 0", "color": "#555"}),
            html.H2(f"{CAUGHT / TOTAL_DEFECTS:.0%}" if TOTAL_DEFECTS else "N/A", style={"margin": 0}),
        ]),
    ]),

    html.Div(style={"display": "flex", "gap": "16px", "marginBottom": "24px"}, children=[
        html.Div(style={"flex": 1, "backgroundColor": "white", "padding": "8px", "borderRadius": "8px",
                         "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"},
                 children=[dcc.Graph(figure=fig_trend)]),
        html.Div(style={"flex": 1, "backgroundColor": "white", "padding": "8px", "borderRadius": "8px",
                         "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"},
                 children=[dcc.Graph(figure=fig_importance)]),
    ]),

    html.Div(style={"backgroundColor": "white", "padding": "16px", "borderRadius": "8px",
                     "boxShadow": "0 1px 3px rgba(0,0,0,0.1)"}, children=[
        html.H3("Flagged Units for Inspection (ranked by risk)"),
        dash_table.DataTable(
            data=flagged_table.to_dict("records"),
            columns=[{"name": c, "id": c} for c in flagged_table.columns],
            style_cell={"fontFamily": "Arial", "padding": "6px", "textAlign": "left"},
            style_header={"fontWeight": "bold", "backgroundColor": "#f0f0f0"},
            style_data_conditional=[
                {"if": {"filter_query": '{actual_defect} = "Defect"'}, "backgroundColor": "#ffe6e6"},
            ],
            page_size=15,
            sort_action="native",
        ),
    ]),
])

if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=8050)
