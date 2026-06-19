import pandas as pd
import datetime
import joblib
import requests
import torch
import torch.nn as nn

# Load trained model and encoders
encoder = joblib.load("encoder.pkl")
scaler = joblib.load("scaler.pkl")

input_dim = len(encoder.get_feature_names_out())
model = nn.Sequential(
    nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.3),
    nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(256, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 1)
)
model.load_state_dict(torch.load("water_requirement_model.pth", map_location=torch.device('cpu')))
model.eval()

# Display possible inputs for each category
print("\nPossible Inputs:")
print("Crop Types:", list(df['CROP TYPE'].unique()))
print("Soil Types:", list(df['SOIL TYPE'].unique()))
print("Regions:", list(df['REGION'].unique()))
print("Weather Conditions:", list(df['WEATHER CONDITION'].unique()))

# Get user input
city = input("Enter the city name in India: ")
crop_type = input("Enter Crop Type: ").strip().upper()
soil_type = input("Enter Soil Type: ").strip().upper()
region = input("Enter Region: ").strip().upper()
weather_condition = input("Enter Weather Condition: ").strip().upper()

# Check feasibility of growing the crop
def is_feasible_time(crop_type):
    crop_calendar = {
        "BANANA": ["Year-round"],
        "SOYABEAN": ["June-July", "September-October"],
        "CABBAGE": ["September-October", "December-March"],
        "POTATO": ["September-November", "January-March"],
        "RICE": ["June-July", "October-November"],
        "MELON": ["January-February", "April-June"],
        "MAIZE": ["June-July", "September-October"],
        "CITRUS": ["February-March", "November-December"],
        "BEAN": ["February-March", "April-May"],
        "WHEAT": ["October-November", "March-April"],
        "MUSTARD": ["October-November", "February-March"],
        "COTTON": ["April-May", "October-November"],
        "SUGARCANE": ["February-March", "December-January (next year)"],
        "TOMATO": ["June-August", "November-December", "October-December", "April-June"],
        "ONION": ["October-November", "May-June", "April-June", "October-November"]
    }

    current_month = datetime.datetime.now().strftime("%B")

    if crop_type not in crop_calendar:
        print("❌ Crop not found in database.")
        return False

    sowing_periods = crop_calendar[crop_type]

    if "Year-round" in sowing_periods:
        print("✅ Feasible time to grow this crop.")
        return True

    is_feasible = any(current_month.strip() in period.strip() for period in sowing_periods)

    if not is_feasible:
        print("❌ Not a feasible time to grow this crop in most regions of India. However, specific conditions may allow for cultivation.")
    else:
        print("✅ Feasible time to grow this crop.")

    return is_feasible

is_feasible_time(crop_type)

# Fetch 5-day weather forecast
def get_weather_forecast(city):
    API_KEY = "ec9f8944c1df4086e1a200f2416cf9cb"  # Replace with your OpenWeatherMap API key
    forecast_url = f"https://api.openweathermap.org/data/2.5/forecast?q={city},IN&appid={API_KEY}&units=metric"
    response = requests.get(forecast_url).json()
    print("\n📅 5-Day Weather Forecast:")
    needs_irrigation = True

    for i in range(0, len(response["list"]), 8):
        forecast = response["list"][i]
        date = forecast["dt_txt"].split()[0]
        temp = forecast["main"]["temp"]
        humidity = forecast["main"]["humidity"]
        rain_prob = forecast.get("rain", {}).get("3h", 0)
        weather_desc = forecast["weather"][0]["description"]

        print(f"{date} -> 🌡 {temp}°C, 💧 {humidity}%, 🌦 {rain_prob} mm, ☁ {weather_desc}")

        if rain_prob > 5:
            needs_irrigation = False

    return needs_irrigation

irrigation_needed = get_weather_forecast(city)

# Prepare input data for prediction
input_data = pd.DataFrame([[crop_type, soil_type, region, weather_condition]],
                          columns=['CROP TYPE', 'SOIL TYPE', 'REGION', 'WEATHER CONDITION'])
input_encoded = encoder.transform(input_data)
input_encoded_df = pd.DataFrame(input_encoded, columns=encoder.get_feature_names_out())

# Ensure column names match training data
missing_cols = set(encoder.get_feature_names_out()) - set(input_encoded_df.columns)
for col in missing_cols:
    input_encoded_df[col] = 0
input_encoded_df = input_encoded_df[encoder.get_feature_names_out()]

# Scale input data
input_scaled = scaler.transform(input_encoded_df)
input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
with torch.no_grad():
    predicted_water = model(input_tensor).numpy().flatten()[0]

# Final recommendation
if irrigation_needed:
    print(f"\n💧 Recommendation: You SHOULD irrigate your field. Predicted Water Requirement: {predicted_water:.2f} mm/ha")
else:
    print("\n🌧 Recommendation: You DO NOT need to irrigate as significant rainfall is expected.")
