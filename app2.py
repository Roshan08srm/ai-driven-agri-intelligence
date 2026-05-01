import streamlit as st
from streamlit_folium import st_folium
import folium
from folium.plugins import LocateControl, AntPath
import requests
import pandas as pd
import numpy as np
import random
import pickle
from PIL import Image
import os
import cv2
import math
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Try importing TensorFlow
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

st.set_page_config(page_title="AI Driven Agri Intelligence", layout="wide", initial_sidebar_state="expanded")

# --- GLOBAL REFRESH FOR REVIEW ---
if 'refresh_done' not in st.session_state:
    st.cache_data.clear()
    st.cache_resource.clear()
    st.session_state.refresh_done = True

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp {
    background: #050B18 !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}

/* Full-width main block and remove top white space */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 0rem !important;
    padding-left: 1rem !important;
    padding-right: 1rem !important;
    max-width: 100% !important;
}

header {
    visibility: hidden !important;
    height: 0 !important;
}

footer {
    visibility: hidden !important;
}

/* ─── SIDEBAR ─── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1628 0%, #060D1A 100%) !important;
    border-right: 1px solid rgba(34, 197, 94, 0.2) !important;
}
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}
[data-testid="stSidebar"] h2 {
    color: #22C55E !important;
    font-size: 0.85rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    padding: 0.5rem 0 0.25rem 0 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stNumberInput label,
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] p {
    color: #CBD5E1 !important;
    font-size: 0.82rem !important;
}
[data-testid="stSidebar"] input,
[data-testid="stSidebar"] select {
    background: #0F2037 !important;
    color: #F1F5F9 !important;
    border: 1px solid #1E3A5F !important;
    border-radius: 8px !important;
}
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, #16A34A, #15803D) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    padding: 0.6rem 1rem !important;
    width: 100% !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    transition: all 0.2s !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, #22C55E, #16A34A) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(34, 197, 94, 0.4) !important;
}

/* ─── MAIN CONTENT TEXT ─── */
.stApp p, .stApp span, .stApp div, .stApp label {
    color: #E2E8F0 !important;
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {
    color: #FFFFFF !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 700 !important;
}

/* ─── TABS ─── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 16px !important;
    padding: 6px !important;
    gap: 4px !important;
    margin-bottom: 1.5rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    border-radius: 12px !important;
    padding: 10px 28px !important;
    border: none !important;
    transition: all 0.25s ease !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #16A34A 0%, #0D9488 100%) !important;
    color: #FFFFFF !important;
    box-shadow: 0 4px 15px rgba(22, 163, 74, 0.35) !important;
}

/* ─── BUTTONS ─── */
.stButton > button {
    background: linear-gradient(135deg, #16A34A 0%, #0D9488 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.04em !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(22, 163, 74, 0.25) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(22, 163, 74, 0.45) !important;
}

/* ─── SLIDER ─── */
.stSlider [data-baseweb="slider"] {
    color: #22C55E !important;
}
.stSlider label {
    color: #94A3B8 !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
}

/* ─── FILE UPLOADER ─── */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.03) !important;
    border: 2px dashed rgba(34, 197, 94, 0.3) !important;
    border-radius: 16px !important;
    padding: 1rem !important;
}
[data-testid="stFileUploader"] * {
    color: #94A3B8 !important;
}

/* ─── SELECT BOXES ─── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}

/* ─── STATUS BADGES ─── */
.stSuccess, .stError, .stWarning, .stInfo {
    border-radius: 12px !important;
    font-weight: 600 !important;
}

/* ─── PROGRESS BAR ─── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #16A34A, #22D3EE) !important;
    border-radius: 99px !important;
}

/* ─── SCROLLBAR ─── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #050B18; }
::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }

/* ─── METRIC FIX ─── */
[data-testid="stMetricValue"] {
    color: #22C55E !important;
    font-size: 2rem !important;
    font-weight: 800 !important;
    line-height: 1 !important;
}
[data-testid="stMetricLabel"] {
    color: #94A3B8 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    opacity: 1 !important;
}

/* ─── EXPANDER ─── */
.streamlit-expanderHeader {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# ==========================================
# 1. API DATA FETCHING (OPEN-METEO)
# ==========================================
def get_open_meteo_data(lat, lon):
    try:
        elev_url = f"https://api.open-meteo.com/v1/elevation?latitude={lat}&longitude={lon}"
        elev_resp = requests.get(elev_url, timeout=3).json()
        elevation = elev_resp.get('elevation', [0])[0]

        weather_url = (
            f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
            "&current=temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m"
            "&daily=precipitation_sum&past_days=3&timezone=auto"
        )
        w_resp = requests.get(weather_url, timeout=3).json()
        
        current = w_resp.get('current', {})
        daily = w_resp.get('daily', {})
        recent_rain_sum = sum(daily.get('precipitation_sum', [0])[:3])

        return {
            'temp': current.get('temperature_2m', 30), 'humidity': current.get('relative_humidity_2m', 60),
            'wind_speed': current.get('wind_speed_10m', 10), 'wind_deg': current.get('wind_direction_10m', 0),
            'elevation': elevation, 'recent_rain_sum': recent_rain_sum, 'status': "✅ Online (Live API)"
        }
    except Exception as e:
        return {'temp': 30, 'humidity': 70, 'wind_speed': 10, 'wind_deg': 0, 'elevation': 50, 'recent_rain_sum': 0, 'status': f"⚠️ Offline Fallback"}

def get_historical_rain(lat, lon):
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
        hist_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&daily=precipitation_sum&timezone=auto"
        resp = requests.get(hist_url, timeout=3).json()
        if 'daily' in resp and 'precipitation_sum' in resp['daily']: return sum([x for x in resp['daily']['precipitation_sum'] if x is not None])
        return 0.0
    except: return 0.0

def get_flood_data(lat, lon):
    try:
        flood_url = f"https://flood-api.open-meteo.com/v1/flood?latitude={lat}&longitude={lon}&daily=river_discharge&forecast_days=1"
        resp = requests.get(flood_url, timeout=3).json()
        if 'daily' in resp and 'river_discharge' in resp['daily']:
            discharge = resp['daily']['river_discharge'][0]
            if discharge is None: return 0.0
            return discharge
        return 0.0
    except: return 0.0

def predict_water_runoff(lat, lon, rain_mm):
    offset = 0.005 
    points = {"Center": (lat, lon), "North": (lat+offset, lon), "South": (lat-offset, lon), "East": (lat, lon+offset), "West": (lat, lon-offset)}
    elevations = {}
    for direction, (p_lat, p_lon) in points.items():
        try: elevations[direction] = requests.get(f"https://api.open-meteo.com/v1/elevation?latitude={p_lat}&longitude={p_lon}", timeout=3).json().get('elevation', [0])[0]
        except: elevations[direction] = 0 
            
    center_elev = elevations["Center"]
    lowest_dir, lowest_elev = "Center", center_elev
    for direction in ["North", "South", "East", "West"]:
        if elevations[direction] < lowest_elev:
            lowest_elev = elevations[direction]; lowest_dir = direction
            
    elevation_drop = center_elev - lowest_elev
    if lowest_dir == "Center" or elevation_drop < 0.5:
        return {"type": "Pool", "target": (lat, lon), "radius": rain_mm * 15}
    return {"type": "Runoff", "target": points[lowest_dir], "radius": rain_mm * 2 * elevation_drop}

def calculate_next_point(lat, lon, angle, distance_km):
    R = 6378.1
    brng = math.radians(angle)
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat1)*math.cos(distance_km/R) + math.cos(lat1)*math.sin(distance_km/R)*math.cos(brng))
    lon2 = lon1 + math.atan2(math.sin(brng)*math.sin(distance_km/R)*math.cos(lat1), math.cos(distance_km/R)-math.sin(lat1)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371 
    dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# ==========================================
# 2. CROWDSOURCED REGIONAL SURVEILLANCE
# ==========================================
def log_regional_threat(lat, lon, disease, transmission_type):
    file = 'regional_threats.csv'
    date_str = datetime.now().strftime("%Y-%m-%d")
    new_data = pd.DataFrame([[date_str, lat, lon, disease, transmission_type]], columns=['Date', 'Lat', 'Lon', 'Disease', 'Type'])
    if os.path.exists(file) and os.path.getsize(file) > 0:
        try:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
            df = df[df['Date'] > (datetime.now() - timedelta(days=7))]
            df = pd.concat([df, new_data], ignore_index=True)
            df.drop_duplicates(subset=['Date', 'Lat', 'Lon', 'Disease'], inplace=True)
        except pd.errors.EmptyDataError:
            df = new_data
    else: df = new_data
    df.to_csv(file, index=False)

    hist_file = 'historical_outbreaks.csv'
    if os.path.exists(hist_file) and os.path.getsize(hist_file) > 0:
        try:
            df_h = pd.read_csv(hist_file)
            if 'Date' in df_h.columns:
                df_h['Date'] = pd.to_datetime(df_h['Date'], format='mixed', errors='coerce')
                df_h = df_h[df_h['Date'] > (datetime.now() - timedelta(days=7))]
            df_h = pd.concat([df_h, new_data], ignore_index=True)
            df_h.drop_duplicates(subset=['Date', 'Lat', 'Lon', 'Disease'], inplace=True)
            df_h.to_csv(hist_file, index=False)
        except pd.errors.EmptyDataError:
            new_data.to_csv(hist_file, index=False)
    else:
        new_data.to_csv(hist_file, index=False)

def get_nearby_threats(lat, lon, radius_km=50):
    if not os.path.exists('regional_threats.csv') or os.path.getsize('regional_threats.csv') == 0: return []
    try:
        df = pd.read_csv('regional_threats.csv')
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        df = df[df['Date'] > (datetime.now() - timedelta(days=7))]
        threats = []
        for _, row in df.iterrows():
            dist = haversine(lat, lon, float(row['Lat']), float(row['Lon']))
            if 0.0 <= dist <= radius_km: 
                # Preserved exact coordinates
                threats.append({'Disease': row['Disease'], 'Type': row['Type'], 'Dist': dist, 'Lat': float(row['Lat']), 'Lon': float(row['Lon'])})
        
        unique_threats = {}
        for t in threats:
            if t['Disease'] not in unique_threats or t['Dist'] < unique_threats[t['Disease']]['Dist']:
                unique_threats[t['Disease']] = t
        return sorted(list(unique_threats.values()), key=lambda x: x['Dist'])
    except: return []

# -------------------------------------------------------------------
# 1. FINAL GRAD-CAM++ ENGINE
# -------------------------------------------------------------------
def make_gradcam_plus_plus(model, img_array, class_index):
    # Retrieve top-level layers dynamically
    vgg = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or layer.name == "vgg16":
            vgg = layer
            break

    if vgg is None: return np.zeros((224, 224))
    
    # target conv layer inside the nested VGG model
    target_layer = vgg.get_layer("block5_conv3")

    # 🔥 Step 1: Build a sub-model that exposes the internal VGG activations
    vgg_grad_model = tf.keras.models.Model(
        inputs=vgg.input,
        outputs=[target_layer.output, vgg.output]
    )

    # Note: We need GAP and Dense for the classification head
    gap = model.get_layer("global_average_pooling2d_4")
    dense = model.get_layer("dense_4")

    with tf.GradientTape() as tape:
        # 🔥 Step 2: Pass through the Sub-Model bridge
        conv_outputs, vgg_output = vgg_grad_model(img_array)
        
        # 🔥 Step 3: Complete the forward pass
        x = gap(vgg_output)
        predictions = dense(x)
        loss = predictions[:, class_index]

    # 🔥 Step 4: Gradients
    grads = tape.gradient(loss, conv_outputs)

    # 🚨 safety check
    if grads is None: return np.zeros((224, 224))
    
    # 🔥 Neural Signal Amplifier (Force signal if gradients are weak)
    if tf.reduce_max(tf.abs(grads)) < 1e-10:
        # Fallback to pure activation saliency if gradients vanish
        weights = tf.reduce_mean(conv_outputs, axis=(1, 2))
    else:
        # ---- Grad-CAM++ math ----
        grads2 = grads ** 2
        grads3 = grads ** 3
        sum_conv = tf.reduce_sum(conv_outputs, axis=(1, 2), keepdims=True)
        alpha_num = grads2
        alpha_denom = 2 * grads2 + grads3 * sum_conv
        alpha_denom = tf.where(alpha_denom != 0, alpha_denom, tf.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        weights = tf.reduce_sum(alphas * tf.nn.relu(grads), axis=(1, 2))

    cam = tf.reduce_sum(weights[:, None, None, :] * conv_outputs, axis=-1)
    heatmap = tf.nn.relu(cam)
    
    # Dynamic Normalization
    h_max = tf.reduce_max(heatmap)
    if h_max > 0:
        heatmap = heatmap / h_max
    
    return heatmap[0].numpy()

# -------------------------------------------------------------------
# 2. ADAPTIVE MASKED BLENDER 
# -------------------------------------------------------------------
def generate_colored_gradcam(original_image, heatmap_2d):
    img_array = np.array(original_image)
    if heatmap_2d is None or np.max(heatmap_2d) == 0: return img_array
    h, w, _ = img_array.shape
    heatmap = cv2.resize(heatmap_2d, (w, h))
    
    # Normalize and Boost
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)
    heatmap = np.power(heatmap, 3)

    # Sharp TOP-K masking
    threshold = np.percentile(heatmap, 85)
    mask = heatmap >= threshold
    if np.sum(mask) < 50:
        threshold = np.percentile(heatmap, 75)
        mask = heatmap >= threshold

    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    overlay = img_bgr.copy()
    blended = cv2.addWeighted(img_bgr, 0.3, heatmap_color, 0.7, 0)
    overlay[mask_3d] = blended[mask_3d]
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

# ==========================================
# 4. MODEL LOADING
# ==========================================
@st.cache_resource
def load_brains():
    brains = {}
    crops = ['Rice', 'Sugarcane', 'Corn', 'Potato', 'Tomato', 'Wheat', 'Cotton', 'Maize', 'Paddy']
    data = []
    for _ in range(5000):
        c = random.choice(crops)
        temp, hum, rain, elev, risk = random.uniform(15, 45), random.uniform(20, 100), random.uniform(0, 150), random.uniform(0, 2000), 0
        if c in ['Rice', 'Paddy'] and rain < 20: risk = 1
        if c == 'Potato' and hum > 80 and temp < 25 and rain > 10: risk = 1
        if hum > 90 and temp > 25: risk = 1
        data.append([c, temp, hum, rain, elev, risk])
        
    df = pd.DataFrame(data, columns=['Crop', 'Temp', 'Hum', 'Rain', 'Elev', 'Risk'])
    le_crop = LabelEncoder()
    df['Crop_Code'] = le_crop.fit_transform(df['Crop'])
    brains['risk'] = RandomForestClassifier(n_estimators=100).fit(df[['Crop_Code', 'Temp', 'Hum', 'Rain', 'Elev']], df['Risk'])
    brains['le_risk_crop'] = le_crop
    
    if os.path.exists("fertilizer_model.pkl"):
        try:
            with open("fertilizer_model.pkl", "rb") as f:
                data = pickle.load(f)
                brains['fert_model'], brains['le_soil'], brains['le_crop'] = data['model'], data['le_soil'], data['le_crop']
        except: pass

    if os.path.exists("fertilizers.csv"):
        try: brains['fert_df'] = pd.read_csv("fertilizers.csv")
        except: brains['fert_df'] = pd.DataFrame()
    else: brains['fert_df'] = pd.DataFrame()

    try:
        if TF_AVAILABLE and os.path.exists("vgg16_final.h5"): brains['cnn'] = tf.keras.models.load_model("vgg16_final.h5")
        if os.path.exists("class_names.txt"):
            with open("class_names.txt", "r") as f: brains['classes'] = f.read().splitlines()
    except: pass
        
    if os.path.exists("Crop_Data.csv"):
         try:
            df_base = pd.read_csv("Crop_Data.csv")
            df_base.columns = [c.strip() for c in df_base.columns]
            if 'label' in df_base.columns: df_base.rename(columns={'label': 'Crop'}, inplace=True)
            unique_crops = sorted(df_base['Crop'].unique())
         except: unique_crops = crops
    else: unique_crops = crops

    df_econ = pd.DataFrame({'Crop': unique_crops, 'Avg_Yield_Tons': np.random.uniform(0.5, 20.0, len(unique_crops)), 'Market_Price': np.random.randint(15000, 120000, len(unique_crops)), 'Spray_Cost': np.random.randint(1000, 5000, len(unique_crops))})
    brains['econ_data'] = df_econ
    return brains

brains = load_brains()

# --- FIX: FORCE FRESH LOAD OF METADATA (Bypasses Cache and CLEANS columns) ---
if os.path.exists("disease_metadata.csv") and os.path.getsize("disease_metadata.csv") > 0:
    try:
        brains['epidem_df'] = pd.read_csv("disease_metadata.csv")
        # This strips out invisible spaces and BOMs like \ufeff so KeyError never triggers
        brains['epidem_df'].columns = [str(c).strip().replace('\ufeff', '') for c in brains['epidem_df'].columns]
    except Exception:
        brains['epidem_df'] = pd.DataFrame()
else:
    brains['epidem_df'] = pd.DataFrame()

# ==========================================
# 5. SESSION STATE
# ==========================================
if 'lat' not in st.session_state: st.session_state.lat = 20.5937
if 'lon' not in st.session_state: st.session_state.lon = 78.9629
for key in ['meteo_data', 'cnn_prediction', 'risk_result', 'heatmap_img']: 
    if key not in st.session_state: st.session_state[key] = None
if 'detected_crop' not in st.session_state: st.session_state.detected_crop = "Rice" 
if 'hist_rain' not in st.session_state: st.session_state.hist_rain = 0.0
if 'flood_discharge' not in st.session_state: st.session_state.flood_discharge = 0.0

if not st.session_state.meteo_data:
    st.session_state.meteo_data = get_open_meteo_data(st.session_state.lat, st.session_state.lon)
    st.session_state.hist_rain = get_historical_rain(st.session_state.lat, st.session_state.lon)
    st.session_state.flood_discharge = get_flood_data(st.session_state.lat, st.session_state.lon)

w = st.session_state.meteo_data

# ==========================================
# 6. SIDEBAR - CONTROLS
# ==========================================
st.sidebar.markdown("<h2>📍 Location Hub</h2>", unsafe_allow_html=True)
loc_mode = st.sidebar.radio("Location Mode:", ["Auto-IP", "Search City", "Manual Coords"], horizontal=True)

if loc_mode == "Search City":
    city_in = st.sidebar.text_input("Enter City Name", "Vijayawada")
    if st.sidebar.button("🔍 Search Location"):
        try:
            r = requests.get(f"https://nominatim.openstreetmap.org/search?q={city_in}&format=json&limit=1", headers={'User-Agent': 'AgriApp/1.0'}).json()
            if r:
                st.session_state.lat, st.session_state.lon = float(r[0]['lat']), float(r[0]['lon'])
                st.session_state.meteo_data = None 
                st.rerun()
        except: pass

elif loc_mode == "Manual Coords":
    c1, c2 = st.sidebar.columns(2)
    new_lat = c1.number_input("Lat", value=st.session_state.lat, format="%.4f")
    new_lon = c2.number_input("Lon", value=st.session_state.lon, format="%.4f")
    if st.sidebar.button("⚙️ Update Coords"):
        st.session_state.lat, st.session_state.lon = new_lat, new_lon
        st.session_state.meteo_data = None
        st.rerun()
else:
    if st.sidebar.button("📍 Auto-Locate Me"):
        try:
            import geocoder
            g = geocoder.ip('me')
            if g.latlng:
                st.session_state.lat, st.session_state.lon = g.latlng
                st.session_state.meteo_data = None
                st.rerun()
        except: pass

# ==========================================
# 2. EPIDEMIOLOGY & TRANSMISSION MATH
# ==========================================
def calculate_bearing(lat1, lon1, lat2, lon2):
    d_lon = math.radians(lon2 - lon1)
    y = math.sin(d_lon) * math.cos(math.radians(lat2))
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(d_lon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def calculate_next_point(lat, lon, brng, d):
    R = 6371
    brng, lat, lon = math.radians(brng), math.radians(lat), math.radians(lon)
    lat2 = math.asin(math.sin(lat)*math.cos(d/R) + math.cos(lat)*math.sin(d/R)*math.cos(brng))
    lon2 = lon + math.atan2(math.sin(brng)*math.sin(d/R)*math.cos(lat), math.cos(d/R)-math.sin(lat)*math.sin(lat2))
    return math.degrees(lat2), math.degrees(lon2)

def predict_water_runoff(lat, lon, rain_mm):
    # Simulated topographical runoff (towards nearest lower elevation/water body proxy)
    target_lat = lat - (0.005 if rain_mm > 20 else 0)
    target_lon = lon + (0.005 if rain_mm > 20 else 0)
    return {"target": [target_lat, target_lon], "radius": rain_mm * 10, "type": "Runoff" if rain_mm > 15 else "Stagnant"}

# ==========================================
# 3. CROWDSOURCED REGIONAL SURVEILLANCE
# ==========================================
def log_regional_threat(lat, lon, disease, transmission_type):
    file = 'regional_threats.csv'
    date_str = datetime.now().strftime("%Y-%m-%d")
    new_data = pd.DataFrame([[date_str, lat, lon, disease, transmission_type]], columns=['Date', 'Lat', 'Lon', 'Disease', 'Type'])
    if os.path.exists(file) and os.path.getsize(file) > 0:
        try:
            df = pd.read_csv(file)
            df = pd.concat([df, new_data], ignore_index=True)
            df.drop_duplicates(subset=['Date', 'Lat', 'Lon', 'Disease'], inplace=True)
        except: df = new_data
    else: df = new_data
    df.to_csv(file, index=False)

def get_nearby_threats(lat, lon, radius_km=50):
    if not os.path.exists('regional_threats.csv'): return []
    try:
        df = pd.read_csv('regional_threats.csv')
        threats = []
        for _, row in df.iterrows():
            dist = haversine(lat, lon, float(row['Lat']), float(row['Lon']))
            if dist <= radius_km:
                t_type = row.get('Type', 'Unknown')
                # Lookup transmission if unknown
                if (t_type == 'Unknown' or pd.isna(t_type)) and 'epidem_df' in brains:
                    m = brains['epidem_df'][brains['epidem_df']['Disease_Keyword'].apply(lambda x: str(x).lower() in row['Disease'].lower())]
                    if not m.empty: t_type = m.iloc[0]['Transmission_Type']
                
                threats.append({'Disease': row['Disease'], 'Type': t_type, 'Dist': dist, 'Lat': float(row['Lat']), 'Lon': float(row['Lon'])})
        return sorted(threats, key=lambda x: x['Dist'])
    except: return []

# ==========================================
# 4. FIELD SENSORS & SIDEBAR
# ==========================================
st.sidebar.markdown("---")
st.sidebar.markdown("<h2>🚜 Field Setup</h2>", unsafe_allow_html=True)
crop_options = list(brains['le_risk_crop'].classes_)
default_idx = crop_options.index(st.session_state.detected_crop) if st.session_state.detected_crop in crop_options else 0
selected_crop = st.sidebar.selectbox("Active Crop", crop_options, index=default_idx)
acres = st.sidebar.number_input("Field Area (Acres)", 1.0, 100.0, 5.0)

st.sidebar.markdown("---")
st.sidebar.markdown("<h2>🌐 Threat Radar</h2>", unsafe_allow_html=True)
nearby_threats = get_nearby_threats(st.session_state.lat, st.session_state.lon)
if nearby_threats:
    st.sidebar.error(f"⚠️ {len(nearby_threats)} Alerts within 50km")
    for t in nearby_threats:
        t_mode = t['Type']
        mode_icon = "🌬️" if t_mode == "Airborne" else ("🌊" if t_mode == "Waterborne" else "🦟")
        st.sidebar.write(f"- 🦠 **{t['Disease'].split('___')[-1].replace('_',' ')}** ({t['Dist']:.1f}km) - {t_mode} {mode_icon}")
        
        # --- ETA LOGIC ADDED HERE ---
        wind_spd = w.get('wind_speed', 10)
        if wind_spd <= 0: wind_spd = 1
        spd_km_day = (wind_spd * 24) * 0.5
        eta_days = t['Dist'] / spd_km_day
        
        if eta_days < 1:
            st.sidebar.caption(f"🚨 **ETA: < 24 Hours** (Critical)")
        else:
            st.sidebar.caption(f"⏳ **ETA: {eta_days:.1f} Days**")
        # ----------------------------
else:
    st.sidebar.success("✅ No regional threats detected.")

st.sidebar.markdown("---")
st.sidebar.markdown("<h2>📡 Sensor Data</h2>", unsafe_allow_html=True)
st.sidebar.info(f"**Connection:** {w['status']}")
st.sidebar.metric("Rain Saturation (30d)", f"{st.session_state.hist_rain:.1f} mm")
st.sidebar.metric("Vector Spread Speed", f"{w['wind_speed'] * 0.5:.1f} km/day", delta=f"{w['wind_deg']}°")

# ==========================================
# 7. MAIN DASHBOARD HEADER
# ==========================================
st.markdown("<h1>🌱 AI Driven Agri Intelligence</h1>", unsafe_allow_html=True)
st.markdown(f"**Crop Profile:** {selected_crop} &nbsp;&nbsp;|&nbsp;&nbsp; **Coordinates:** {st.session_state.lat:.4f}, {st.session_state.lon:.4f}")

with st.container():
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("🌡️ Temp", f"{w['temp']}°C")
    c2.metric("💧 Humidity", f"{w['humidity']}%")
    c3.metric("🌬️ Wind", f"{w['wind_speed']} km/h")
    c4.metric("🌧️ Rain (3d)", f"{w['recent_rain_sum']} mm")
    c5.metric("⛰️ Elev", f"{w['elevation']:.0f} m")

st.divider()

# ==========================================
# 8. TABS
# ==========================================
t1, t2 = st.tabs(["🌍 Predictive Risk Map", "🔬 Leaf Disease Scanner"])

# TAB 1: RISK MAP 
with t1:
    c_ctrl, c_map = st.columns([1, 3])
    with c_ctrl:
        st.markdown("### ⚙️ Simulation Controls")
        forecast_days = 3 # Fixed 3-day projection

        if st.button("🚀 Run Area Scan", type="primary"):
            c_code = brains['le_risk_crop'].transform([selected_crop])[0]
            prob = brains['risk'].predict_proba([[c_code, w['temp'], w['humidity'], w['recent_rain_sum'], w['elevation']]])[0][1]
            reasons = []
            if prob > 0.5: reasons.append(f"Environmental match for {selected_crop} pathogens.")
            if w['recent_rain_sum'] > 50: prob += 0.2; reasons.append(f"Heavy Rainfall Vectors Active ({w['recent_rain_sum']:.0f}mm)")
            if st.session_state.cnn_prediction and "Healthy" not in st.session_state.cnn_prediction:
                prob = 1.0; reasons.insert(0, f"BIOMETRIC CONFIRMATION: {st.session_state.cnn_prediction}")
            
            st.session_state.risk_result = {"score": int(min(prob, 1.0)*100), "reasons": reasons}
            
        if st.session_state.risk_result:
            val = st.session_state.risk_result['score']
            st.markdown("### ⚠️ Risk Level")
            st.progress(val / 100.0)
            st.metric("Probability", f"{val}%")
            
            if val > 70: st.error("🚨 CRITICAL OUTBREAK RISK")
            elif val > 40: st.warning("⚠️ ELEVATED RISK")
            else: st.success("✅ OPTIMAL CONDITIONS")
            
            st.markdown("#### 🔍 Outbreak Factors")
            for r in st.session_state.risk_result['reasons']: 
                st.info(f"👉 {r}")

    with c_map:
        m = folium.Map([st.session_state.lat, st.session_state.lon], zoom_start=13, tiles="OpenStreetMap")
        folium.LayerControl().add_to(m)
        LocateControl().add_to(m)
        
        folium.Marker([st.session_state.lat, st.session_state.lon], icon=folium.Icon(color="green", icon="leaf"), popup="Your Farm").add_to(m)
        
        # 🔥 50KM BIOSECURITY PERIMETER
        folium.Circle(
            [st.session_state.lat, st.session_state.lon],
            radius=50000, # 50km in meters
            color="#f1c40f",
            fill=True,
            fill_opacity=0.1,
            weight=2,
            dash_array='5, 5',
            popup="50km Biosecurity Perimeter"
        ).add_to(m)
        
        for threat in nearby_threats:
            t_mode = threat['Type']
            m_color = "red" if t_mode == "Airborne" else ("blue" if t_mode == "Waterborne" else "orange")
            folium.Marker([threat['Lat'], threat['Lon']], tooltip=f"Alert: {threat['Disease']} ({t_mode})", icon=folium.Icon(color=m_color, icon='info-sign')).add_to(m)

        if st.session_state.cnn_prediction and "Healthy" not in st.session_state.cnn_prediction:
            raw_pred = st.session_state.cnn_prediction
            d_type, d_mult = "Airborne", 0.5
            
            if 'epidem_df' in brains and not brains['epidem_df'].empty:
                ep_match = brains['epidem_df'][brains['epidem_df']['Disease_Keyword'].apply(lambda x: str(x).lower() in raw_pred.lower())]
                if not ep_match.empty:
                    d_type = ep_match.iloc[0]['Transmission_Type']
                    d_mult = float(ep_match.iloc[0]['Danger_Multiplier'])

            if d_type == "Airborne":
                spread_km = (w['wind_speed'] / 10) * d_mult * forecast_days
                t_lat, t_lon = calculate_next_point(st.session_state.lat, st.session_state.lon, w['wind_deg'], spread_km)
                AntPath([[st.session_state.lat, st.session_state.lon], [t_lat, t_lon]], color='#ef4444', weight=5).add_to(m)
                folium.Circle([t_lat, t_lon], radius=spread_km*500, color="#ef4444", fill=True, opacity=0.3).add_to(m)
            elif d_type == "Waterborne":
                runoff = predict_water_runoff(st.session_state.lat, st.session_state.lon, w['recent_rain_sum'])
                AntPath([[st.session_state.lat, st.session_state.lon], runoff['target']], color='#3b82f6', weight=5).add_to(m)
                folium.Circle(runoff['target'], radius=runoff['radius']*100, color="#3b82f6", fill=True, opacity=0.3).add_to(m)
            else: # Vector
                folium.Circle([st.session_state.lat, st.session_state.lon], radius=1000 * d_mult * forecast_days, color="#f59e0b", fill=True, opacity=0.3).add_to(m)

        # Interactive Map Capture
        m.add_child(folium.LatLngPopup())
        
        map_data = st_folium(m, height=500, use_container_width=True, key="risk_map")
        
        # 🔥 PINPOINT POSITIONING ENGINE (Auto-Scan on Click)
        if map_data and map_data.get("last_clicked"):
            new_lat = map_data["last_clicked"]["lat"]
            new_lon = map_data["last_clicked"]["lng"]
            if (new_lat != st.session_state.lat or new_lon != st.session_state.lon):
                st.session_state.lat = new_lat
                st.session_state.lon = new_lon
                st.session_state.meteo_data = None # Force weather update
                
                # --- AUTO-SCAN LOGIC ---
                try:
                    # Update local weather for new point before scanning
                    w_new = get_sensor_data(new_lat, new_lon)
                    c_code = brains['le_risk_crop'].transform([selected_crop])[0]
                    prob = brains['risk'].predict_proba([[c_code, w_new['temp'], w_new['humidity'], w_new['recent_rain_sum'], w_new['elevation']]])[0][1]
                    reasons = ["Instant Point-Scan Analysis Complete."]
                    if prob > 0.5: reasons.append(f"Environmental match for {selected_crop} pathogens.")
                    if w_new['recent_rain_sum'] > 50: prob += 0.2; reasons.append(f"Heavy Rainfall Vectors detected.")
                    st.session_state.risk_result = {"score": int(min(prob, 1.0)*100), "reasons": reasons}
                except: pass
                # -----------------------
                
                st.rerun()

# TAB 2: DISEASE CAM & AUTO FERTILIZER
with t2:
    col_upload, col_result = st.columns(2)
    with col_upload:
        st.markdown("### 📸 Leaf Analysis")
        unique_crops_found = sorted(list({c.split("___")[0].replace("_", " ").strip() for c in brains.get('classes', [])}))
        cam_crop_filter = st.selectbox("Select Crop Type:", unique_crops_found if unique_crops_found else ["Offline"])

        img_file = st.file_uploader("Upload Leaf Image", type=['jpg','png'])
        if img_file and 'cnn' in brains:
            img = Image.open(img_file).convert('RGB')
            st.image(img, caption="Original Image", use_container_width=True)
            
            if st.button("🔍 Scan for Diseases", type="primary"):
                img_arr = tf.keras.preprocessing.image.img_to_array(img.resize((224,224)))
                img_arr_expanded = np.expand_dims(img_arr, 0)/255.0
                
                with st.spinner("Analyzing Pathogen Features..."):
                    # Use official VGG16 preprocessor
                    img_array_vgg = tf.keras.applications.vgg16.preprocess_input(img_arr.copy().reshape(1, 224, 224, 3))
                    preds = brains['cnn'].predict(img_array_vgg)[0]
                    pred_idx = np.argmax(preds)
                    predicted_class = brains['classes'][pred_idx]
                    
                    # 🔥 Robust Crop Extraction (Check prefix and parentheses)
                    predicted_crop = predicted_class.split("___")[0].replace("_", " ").strip()
                    if "(" in predicted_class:
                        alt_crop = predicted_class.split("(")[-1].split(")")[0].strip()
                        if alt_crop.lower() in ["rice", "wheat", "corn", "maize", "sugarcane", "cotton", "potato", "tomato"]:
                            predicted_crop = alt_crop

                    # Permissive Matching (Allow 'Unknown' or direct match)
                    is_match = (predicted_crop.lower().replace(" ", "") == cam_crop_filter.lower().replace(" ", "")) or ("unknown" in predicted_crop.lower())
                    
                    if is_match:
                        st.session_state.cnn_prediction = predicted_class
                        
                        # 🔥 Second Opinion Engine: Always run Grad-CAM++ to double-check 'Healthy' labels
                        raw_heatmap = make_gradcam_plus_plus(brains['cnn'], img_array_vgg, pred_idx)
                        st.session_state.heatmap_img = generate_colored_gradcam(img, raw_heatmap)
                        
                        # 🚨 Threshold Override: If 'Healthy' but engine finds strong localized spots
                        if "healthy" in predicted_class.lower() and np.max(raw_heatmap) > 0.7:
                            st.session_state.cnn_prediction = "Unknown Disease (Anomalous Signature)"
                            st.warning("⚠️ SECOND OPINION: Biometric engine detected anomalous localized signatures. This may be an early-stage pathogen or misclassification.")
                        
                        if "healthy" in predicted_class.lower():
                            st.success(f"✅ Analysis Complete: Healthy {predicted_crop} leaf detected.")
                        else:
                            st.success("✅ Analysis Complete: Pathogen detected.")
                        
                        if "healthy" not in predicted_class.lower():
                            log_type = "Unknown"
                            if not brains['epidem_df'].empty and 'Disease_Keyword' in brains['epidem_df'].columns:
                                m_type = brains['epidem_df'][brains['epidem_df']['Disease_Keyword'].apply(lambda x: "".join(c for c in str(x).lower() if c.isalnum()) in "".join(c for c in st.session_state.cnn_prediction.lower() if c.isalnum()))]
                                if not m_type.empty: log_type = m_type.iloc[0]['Transmission_Type']
                            log_regional_threat(st.session_state.lat, st.session_state.lon, st.session_state.cnn_prediction, log_type)
                    else:
                        st.warning(f"⚠️ Warning: Image is not recognized as {cam_crop_filter}. It appears to be a {predicted_crop} leaf.")
                        st.session_state.cnn_prediction = None
                        st.session_state.heatmap_img = None

    with col_result:
        st.markdown("### 🧬 AI Diagnostics")
        if st.session_state.cnn_prediction:
            raw_name = st.session_state.cnn_prediction
            display_name = raw_name.replace("___", " : ").replace("_", " ").title()
            
            d_type_display = "UNKNOWN"
            if not brains['epidem_df'].empty and 'Disease_Keyword' in brains['epidem_df'].columns:
                m_type = brains['epidem_df'][brains['epidem_df']['Disease_Keyword'].apply(lambda x: "".join(c for c in str(x).lower() if c.isalnum()) in "".join(c for c in raw_name.lower() if c.isalnum()))]
                if not m_type.empty: 
                    t_val = m_type.iloc[0]['Transmission_Type']
                    d_type_display = str(t_val).upper() if pd.notna(t_val) and str(t_val).lower() != 'nan' else "UNKNOWN"
            
            st.markdown(f"**Diagnosis:** {display_name}")
            
            if "healthy" in raw_name.lower(): 
                st.success("✅ Plant appears perfectly healthy.")
                st.info("No transmission risk.")
            else:
                st.info(f"**Transmission Type:** {d_type_display}")
                st.error("⚠️ Pathogen Confirmed")
                if st.session_state.heatmap_img is not None:
                    st.image(st.session_state.heatmap_img, caption="Grad-CAM++ Pathogen Map", use_container_width=True)
        else:
            st.write("Awaiting image upload...")