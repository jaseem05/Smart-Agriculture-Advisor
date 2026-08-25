from pathlib import Path
from flask import Flask, render_template, request
import pickle
import pandas as pd
import plotly.express as px

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

MONTHS = [
    "January",
    "February",
    "March",
    "April",
    "May",
    "June",
    "July",
    "August",
    "September",
    "October",
    "November",
    "December"
]

SOIL_TYPES = ["Loamy", "Clay", "Sandy", "Black"]
DEMAND_LEVELS = ["High", "Medium", "Low"]
STATES = [
    "Andhra Pradesh",
    "Gujarat",
    "Karnataka",
    "Kerala",
    "Maharashtra",
    "Punjab",
    "Rajasthan",
    "Tamil Nadu",
    "Telangana",
    "Uttar Pradesh"
]

RICE_PRICE_RULES = {
    "High": (53, 53),
    "Medium": (48, 52),
    "Low": (43, 48)
}


def load_pickle(filename, required=True):
    file_path = BASE_DIR / filename
    if not file_path.exists():
        if required:
            raise FileNotFoundError(
                f"Required file '{filename}' was not found in {BASE_DIR}"
            )
        return None

    with file_path.open("rb") as model_file:
        return pickle.load(model_file)


# Load models from the same folder as this app, even if Flask is started elsewhere.
crop_model = load_pickle("crop_recommendation_model.pkl", required=False)
price_model = load_pickle("xgboost_market_price_model.pkl")

# Load dataset
df = pd.read_csv(BASE_DIR / "market_price_dataset_2025.csv")


def fallback_crop_recommendation(month_encoded, temperature, rainfall, soil):
    if rainfall >= 150 or (temperature >= 25 and 5 <= month_encoded <= 8):
        return "Rice"
    if temperature <= 22 and month_encoded in (10, 11, 0, 1):
        return "Wheat"
    if soil in ("Loamy", "Clay") and temperature <= 25:
        return "Potato"
    if 20 <= temperature <= 30 and rainfall <= 120:
        return "Tomato"
    return "Corn"


def crop_advice(crop_name, temperature, rainfall, soil, predicted_price):
    crop_tips = {
        "Rice": [
            "Maintain steady water availability during early growth.",
            "Use level fields and avoid water stress during transplanting.",
            "Monitor for fungal issues when humidity and rainfall are high."
        ],
        "Tomato": [
            "Use staking or support to improve fruit quality.",
            "Keep irrigation consistent to reduce fruit cracking.",
            "Watch for leaf curl and blight during warm, humid periods."
        ],
        "Potato": [
            "Use loose, well-drained soil for better tuber formation.",
            "Avoid waterlogging because it can cause tuber rot.",
            "Begin earthing up when plants reach active vegetative growth."
        ],
        "Wheat": [
            "Prefer cool and dry conditions during grain filling.",
            "Avoid excess irrigation close to maturity.",
            "Check for rust disease if the weather turns humid."
        ],
        "Corn": [
            "Ensure good spacing because maize is sensitive to crowding.",
            "Keep moisture stable during tasseling and grain formation.",
            "Apply nitrogen in split doses for stronger growth."
        ]
    }

    alerts = []
    if rainfall < 50:
        alerts.append("Rainfall is low, so plan irrigation before sowing.")
    elif rainfall > 180:
        alerts.append("Rainfall is high, so drainage should be checked.")

    if temperature > 34:
        alerts.append("Temperature is high, so use mulching and avoid midday irrigation.")
    elif temperature < 16:
        alerts.append("Temperature is low, so choose a protected sowing window.")

    soil_tip = f"{soil} soil is selected. Add organic matter to improve nutrient holding and root growth."

    if predicted_price >= 70:
        market_tip = "Market signal looks strong. Compare nearby mandi prices before selling."
    elif predicted_price >= 40:
        market_tip = "Market signal is moderate. Stagger selling if storage is available."
    else:
        market_tip = "Market signal is weak. Consider storage, processing, or alternate local demand."

    return {
        "crop_tips": crop_tips.get(crop_name, ["Use local agronomy guidance before sowing."]),
        "alerts": alerts or ["Weather inputs look suitable for a normal cultivation plan."],
        "soil_tip": soil_tip,
        "market_tip": market_tip
    }


def apply_rice_price_rule(predicted_price, demand):
    min_price, max_price = RICE_PRICE_RULES[demand]
    if min_price == max_price:
        return float(min_price)
    return round(max(min_price, min(predicted_price, max_price)), 2)


def predict_market_price(crop_name, month, month_encoded, demand, state):
    crop_price_dict = {name: index for index, name in enumerate(sorted(df["Crop_Name"].unique()))}
    month_price_dict = {name: index for index, name in enumerate(sorted(df["Month"].unique()))}
    demand_dict = {name: index for index, name in enumerate(sorted(df["Demand"].unique()))}
    state_dict = {name: index for index, name in enumerate(sorted(df["State"].unique()))}

    matching_prices = df[
        (df["Crop_Name"] == crop_name)
        & (df["Month"] == month)
        & (df["Demand"] == demand)
        & (df["State"] == state)
    ]["Market_Price_per_kg"]

    if not matching_prices.empty:
        predicted_price = round(matching_prices.mean(), 2)
    else:
        price_prediction = price_model.predict([[
            crop_price_dict.get(crop_name, 0),
            month_price_dict.get(month, month_encoded),
            demand_dict[demand],
            state_dict[state]
        ]])
        predicted_price = round(price_prediction[0], 2)

    if crop_name == "Rice":
        predicted_price = apply_rice_price_rule(predicted_price, demand)

    return predicted_price


# HOME PAGE
@app.route("/")
def home():
    return render_template(
        "home.html",
        months=MONTHS,
        soil_types=SOIL_TYPES,
        demand_levels=DEMAND_LEVELS,
        states=STATES
    )


# PREDICTION PAGE
@app.route("/predict", methods=["POST"])
def predict():

    month = request.form["month"]
    soil = request.form["soil"]
    demand = request.form["demand"]
    state = request.form["state"]
    temperature = float(request.form["temperature"])
    rainfall = float(request.form["rainfall"])

    # Encoding manually
    month_dict = {
        "January": 0,
        "February": 1,
        "March": 2,
        "April": 3,
        "May": 4,
        "June": 5,
        "July": 6,
        "August": 7,
        "September": 8,
        "October": 9,
        "November": 10,
        "December": 11
    }

    soil_dict = {
        "Loamy": 0,
        "Clay": 1,
        "Sandy": 2,
        "Black": 3
    }

    month_encoded = month_dict[month]
    soil_encoded = soil_dict[soil]

    crop_names = {
        0: "Rice",
        1: "Tomato",
        2: "Potato",
        3: "Wheat",
        4: "Corn"
    }

    if crop_model is None:
        crop_name = fallback_crop_recommendation(
            month_encoded,
            temperature,
            rainfall,
            soil
        )
    else:
        crop_prediction = crop_model.predict([[
            month_encoded,
            temperature,
            rainfall,
            soil_encoded,
            0
        ]])
        crop_name = crop_names.get(int(crop_prediction[0]), str(crop_prediction[0]))

    # Market price prediction
    predicted_price = predict_market_price(crop_name, month, month_encoded, demand, state)

    # Harvest prediction (simple demo)
    harvest_days = 90
    advice = crop_advice(crop_name, temperature, rainfall, soil, predicted_price)

    return render_template(
        "result.html",
        crop=crop_name,
        price=predicted_price,
        harvest=harvest_days,
        month=month,
        soil=soil,
        demand=demand,
        state=state,
        temperature=temperature,
        rainfall=rainfall,
        advice=advice
    )


# DASHBOARD PAGE
@app.route("/dashboard")
def dashboard():
    avg_price = round(df["Market_Price_per_kg"].mean(), 2)
    top_crop = df.groupby("Crop_Name")["Market_Price_per_kg"].mean().idxmax()
    top_state = df.groupby("State")["Market_Price_per_kg"].mean().idxmax()
    crop_price_summary = (
        df.groupby("Crop_Name", as_index=False)["Market_Price_per_kg"]
        .mean()
        .sort_values("Market_Price_per_kg", ascending=False)
    )
    crop_price_summary["Market_Price_per_kg"] = crop_price_summary["Market_Price_per_kg"].round(2)
    demand_summary = df["Demand"].value_counts().reset_index()
    demand_summary.columns = ["Demand", "Count"]

    crop_bar = px.bar(
        crop_price_summary,
        x="Crop_Name",
        y="Market_Price_per_kg",
        color="Crop_Name",
        text="Market_Price_per_kg",
        title="Average Market Price by Crop",
        labels={
            "Crop_Name": "Crop",
            "Market_Price_per_kg": "Average Price per kg"
        }
    )
    crop_bar.update_traces(texttemplate="Rs. %{text}", textposition="outside")
    crop_bar.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=80),
        height=520,
        xaxis_tickangle=-35
    )

    demand_bar = px.bar(
        demand_summary,
        x="Demand",
        y="Count",
        text="Count",
        title="Demand Level Count",
        color="Demand",
        color_discrete_map={
            "High": "#2f6f4e",
            "Medium": "#c98432",
            "Low": "#7f8a82"
        },
        labels={"Demand": "Demand", "Count": "Records"}
    )
    demand_bar.update_traces(textposition="outside")
    demand_bar.update_layout(
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=60),
        height=420
    )

    crop_graph = crop_bar.to_html(full_html=False)
    demand_graph = demand_bar.to_html(full_html=False)

    return render_template(
        "dashboard.html",
        crop_graph=crop_graph,
        demand_graph=demand_graph,
        avg_price=avg_price,
        top_crop=top_crop,
        top_state=top_state
    )


if __name__ == "__main__":
    app.run(debug=True)