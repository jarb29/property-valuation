import streamlit as st
import pandas as pd
import time
from pathlib import Path
import sys
import importlib.util

# Setup paths
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent / "src"))

# Page config
st.set_page_config(
    page_title="PropertyAI - Real Estate Valuation",
    page_icon="🏠",
    layout="wide"
)

# Import model
spec = importlib.util.spec_from_file_location("predictor", current_dir / "models" / "predictor.py")
predictor_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(predictor_module)
PropertyPredictor = predictor_module.PropertyPredictor

# Utility functions
def format_currency(amount):
    return f"${amount:,.0f} CLP"

def get_sector_coordinates():
    return {
        'las condes': {'lat': -33.4172, 'lng': -70.5476},
        'vitacura': {'lat': -33.3900, 'lng': -70.5700},
        'lo barnechea': {'lat': -33.3500, 'lng': -70.5000},
        'nunoa': {'lat': -33.4569, 'lng': -70.5975},
        'providencia': {'lat': -33.4372, 'lng': -70.6178},
        'la reina': {'lat': -33.4450, 'lng': -70.5350}
    }

@st.cache_resource
def load_model():
    try:
        return PropertyPredictor()
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def main():
    # Compact CSS styling
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0.8rem;
        border-radius: 8px;
        text-align: center;
        margin-bottom: 0.8rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    .section-container {
        background: white;
        padding: 0.8rem;
        border-radius: 6px;
        box-shadow: 0 1px 5px rgba(0,0,0,0.05);
        border: 1px solid #e0e0e0;
        margin-bottom: 0.5rem;
    }
    .stSelectbox > div > div {
        background-color: #f8f9fa;
    }
    .stNumberInput > div > div > input {
        background-color: #f8f9fa;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Compact header
    st.markdown("""
    <div class="main-header">
        <h2 style="color: white; margin: 0; font-size: 1.8rem;">🏠 PropertyAI</h2>
        <p style="color: #e8f4fd; margin: 0.2rem 0 0 0; font-size: 0.9rem;">Real Estate Valuation Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    predictor = load_model()
    if not predictor:
        st.error("❌ Model could not be loaded.")
        return
    
    # Tabs
    tab1, tab2 = st.tabs(["🏠 Property Valuation", "📊 Model Information"])
    
    with tab1:
        # Two column layout
        left_col, right_col = st.columns([1, 1])
        
        with left_col:
            st.markdown("""
            <div class="section-container">
                <h4 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.1rem;">📋 Property Information</h4>
            """, unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                property_type = st.selectbox(
                    "Property Type",
                    ["departamento", "casa"],
                    format_func=lambda x: "🏢 Apartment" if x == "departamento" else "🏡 House"
                )
            
            with col2:
                sector = st.selectbox(
                    "Location Sector",
                    ["las condes", "vitacura", "lo barnechea", "nunoa", "providencia", "la reina"],
                    format_func=lambda x: f"📍 {x.replace('_', ' ').title()}"
                )
            
            with col3:
                n_rooms = st.number_input(
                    "Bedrooms",
                    min_value=1,
                    max_value=10,
                    value=3
                )
            
            with col4:
                n_bathroom = st.number_input(
                    "Bathrooms",
                    min_value=1,
                    max_value=5,
                    value=2
                )
            
            st.markdown("<div style='margin: 0.8rem 0 0.3rem 0; font-size: 0.9rem; color: #34495e;'><b>📏 Areas & Coordinates</b></div>", unsafe_allow_html=True)
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                net_usable_area = st.number_input(
                    "Usable Area (m²)",
                    min_value=10.0,
                    max_value=1000.0,
                    value=100.0,
                    step=5.0
                )
            
            with col6:
                net_area = st.number_input(
                    "Total Built Area (m²)",
                    min_value=net_usable_area,
                    max_value=2000.0,
                    value=max(120.0, net_usable_area * 1.2),
                    step=5.0
                )
            
            sector_coords = get_sector_coordinates()
            default_coords = sector_coords.get(sector, {"lat": -33.4372, "lng": -70.5476})
            
            with col7:
                latitude = st.number_input(
                    "Latitude",
                    min_value=-33.525,
                    max_value=-33.305,
                    value=default_coords["lat"],
                    step=0.0001,
                    format="%.4f"
                )
            
            with col8:
                longitude = st.number_input(
                    "Longitude",
                    min_value=-70.644,
                    max_value=-70.432,
                    value=default_coords["lng"],
                    step=0.0001,
                    format="%.4f"
                )
            
            st.markdown("<div style='margin: 0.8rem 0 0.3rem 0; font-size: 0.9rem; color: #34495e;'><b>🗺️ Map</b></div>", unsafe_allow_html=True)
            
            try:
                import folium
                from streamlit_folium import st_folium
                
                # Create folium map with only OpenStreetMap
                m = folium.Map(
                    location=[latitude, longitude], 
                    zoom_start=12,
                    tiles='OpenStreetMap'
                )
                
                # Add sector boundaries as circles
                sector_colors = {
                    'las condes': '#FF6B6B',
                    'vitacura': '#4ECDC4', 
                    'lo barnechea': '#45B7D1',
                    'nunoa': '#96CEB4',
                    'providencia': '#FFEAA7',
                    'la reina': '#DDA0DD'
                }
                
                for sect, coords in get_sector_coordinates().items():
                    color = sector_colors.get(sect, '#95A5A6')
                    is_selected = sect == sector
                    
                    folium.Circle(
                        location=[coords['lat'], coords['lng']],
                        radius=2500,
                        popup=f"<b>{sect.replace('_', ' ').title()}</b><br>Premium Santiago sector",
                        color=color,
                        weight=5 if is_selected else 3,
                        fillColor=color,
                        fillOpacity=0.4 if is_selected else 0.2,
                        opacity=1.0
                    ).add_to(m)
                    
                    # Add sector labels
                    folium.Marker(
                        [coords['lat'], coords['lng']],
                        popup=f"<b>{sect.replace('_', ' ').title()}</b>",
                        icon=folium.DivIcon(
                            html=f'<div style="font-size: 10px; color: {color}; font-weight: bold;">{sect.replace("_", " ").title()}</div>',
                            icon_size=(80, 20),
                            icon_anchor=(40, 10)
                        )
                    ).add_to(m)
                
                # Add property marker
                folium.Marker(
                    [latitude, longitude],
                    popup=f"""<div style='width: 200px;'>
                        <h4>🏠 Property Details</h4>
                        <b>Type:</b> {property_type.replace('departamento', 'Apartment').replace('casa', 'House')}<br>
                        <b>Sector:</b> {sector.replace('_', ' ').title()}<br>
                        <b>Area:</b> {net_usable_area} m²<br>
                        <b>Rooms:</b> {n_rooms} BR / {n_bathroom} BA<br>
                        <b>Coordinates:</b> {latitude:.4f}, {longitude:.4f}<br>
                    </div>""",
                    tooltip=f"🏠 Property Location",
                    icon=folium.Icon(
                        color='red', 
                        icon='home',
                        prefix='fa'
                    )
                ).add_to(m)
                
                # Add fullscreen button
                from folium.plugins import Fullscreen
                Fullscreen().add_to(m)
                
                # Display compact map
                st_folium(m, width=700, height=250)
                
            except ImportError:
                st.info(f"📍 **Selected Location**: {sector.replace('_', ' ').title()} sector at coordinates ({latitude:.4f}, {longitude:.4f})")
                st.warning("Install folium and streamlit-folium for interactive map: pip install folium streamlit-folium")
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close section container
            
            predict_button = st.button(
                "🚀 Calculate Property Value",
                type="primary",
                use_container_width=True,
                help="Click to get AI-powered property valuation"
            )
        
        with right_col:
            st.markdown("""
            <div class="section-container">
                <h4 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.1rem;">💰 Valuation Results</h4>
            """, unsafe_allow_html=True)
            
            if predict_button:
                features = {
                    "type": property_type,
                    "sector": sector,
                    "net_usable_area": net_usable_area,
                    "net_area": net_area,
                    "n_rooms": n_rooms,
                    "n_bathroom": n_bathroom,
                    "latitude": latitude,
                    "longitude": longitude
                }
                
                with st.spinner("🔄 Analyzing property data..."):
                    try:
                        start_time = time.time()
                        prediction = predictor.predict(features)
                        prediction_time = time.time() - start_time
                        
                        # Success message
                        st.success("✅ Valuation Complete!")
                        
                        # Ultra-compact main result
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 0.6rem; border-radius: 6px; text-align: center; margin: 0.3rem 0;">
                            <div style="color: white; font-size: 0.9rem; margin: 0;">💎 ESTIMATED VALUE</div>
                            <div style="color: #ffd700; margin: 0.1rem 0; font-size: 1.4rem; font-weight: bold;">{format_currency(prediction)}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Compact metrics in 2x2 grid
                        price_per_m2 = prediction / net_usable_area
                        total_price_per_m2 = prediction / net_area
                        model_info = predictor.get_model_info()
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Price/m²", format_currency(price_per_m2), delta=None)
                            st.metric("Time", f"{prediction_time*1000:.1f}ms", delta=None)
                        with col2:
                            st.metric("Price/Total m²", format_currency(total_price_per_m2), delta=None)
                            st.metric("Model", model_info.get('model_type', 'Unknown'), delta=None)
                        
                        # Ultra-compact summary
                        property_size = "Large" if net_usable_area > 150 else "Medium" if net_usable_area > 80 else "Compact"
                        st.markdown(f"""
                        <div style="background: #f8f9fa; padding: 0.3rem; border-radius: 4px; font-size: 0.8rem; margin: 0.3rem 0;">
                            <b>🏠</b> {property_type.replace('departamento', 'Apt').replace('casa', 'House')} • {sector.replace('_', ' ').title()} • {n_rooms}BR/{n_bathroom}BA • {net_usable_area}m²
                        </div>
                        <div style="background: #fff3cd; padding: 0.3rem; border-radius: 4px; font-size: 0.7rem; margin: 0.3rem 0;">
                            <b>⚠️</b> AI estimate - actual value may vary
                        </div>
                        """, unsafe_allow_html=True)
                            
                    except Exception as e:
                        st.error(f"❌ Prediction failed: {str(e)}")
            else:
                st.markdown("""
                <div style="background: linear-gradient(135deg, #e3f2fd 0%, #bbdefb 100%); border: 1px solid #90caf9; border-radius: 6px; padding: 1rem; text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                    <div style="color: #1565c0; font-size: 1rem; font-weight: bold;">Ready for Valuation</div>
                    <div style="color: #1976d2; font-size: 0.8rem; margin: 0.2rem 0 0 0;">Complete details and click calculate</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)  # Close section container
    
    with tab2:
        st.markdown("""
        <div class="section-container">
            <h4 style="color: #2c3e50; margin: 0 0 0.5rem 0; font-size: 1.1rem;">📊 Model Information</h4>
        """, unsafe_allow_html=True)
        
        try:
            model_info = predictor.get_model_info()
            
            # Compact performance metrics
            metrics = model_info.get('metrics', {})
            if metrics:
                col1, col2, col3 = st.columns(3)
                with col1:
                    if 'rmse' in metrics:
                        st.metric("RMSE", f"{metrics['rmse']:.2f}", delta=None)
                with col2:
                    if 'mae' in metrics:
                        st.metric("MAE", f"{metrics['mae']:.2f}", delta=None)
                with col3:
                    if 'mape' in metrics:
                        st.metric("MAPE", f"{metrics['mape']:.4f}", delta=None)
            
            # Compact model details
            st.markdown(f"""
            <div style="background: #f8f9fa; padding: 0.5rem; border-radius: 5px; font-size: 0.8rem; margin: 0.5rem 0;">
                <b>🤖 Model:</b> {model_info.get('model_type', 'Unknown')} • 
                <b>⚙️ Engine:</b> {model_info.get('inference_engine', 'Unknown')} • 
                <b>📅 Updated:</b> {model_info.get('timestamp', 'Unknown')[:10]}
            </div>
            """, unsafe_allow_html=True)
            
            # Compact features list
            features = model_info.get('features', [])
            if features:
                features_str = ' • '.join([f.replace('_', ' ').title() for f in features])
                st.markdown(f"""
                <div style="background: #e8f5e8; padding: 0.5rem; border-radius: 5px; font-size: 0.8rem; margin: 0.3rem 0;">
                    <b>📊 Features:</b> {features_str}
                </div>
                """, unsafe_allow_html=True)
            
            # Technical details
            st.markdown(f"""
            <div style="background: #fff3e0; padding: 0.5rem; border-radius: 5px; font-size: 0.75rem; margin: 0.3rem 0; line-height: 1.4;">
                <b>🔧 Technical Details:</b><br>
                <b>Preprocessor Path:</b> {model_info.get('preprocessor_path', 'N/A')}<br>
                <b>ONNX Model Path:</b> {model_info.get('onnx_model_path', 'N/A')}<br>
                <b>Original Model Path:</b> {model_info.get('original_model_path', 'N/A')}<br>
                <b>Model Type:</b> {model_info.get('model_type', 'N/A')}<br>
                <b>Inference Engine:</b> {model_info.get('inference_engine', 'N/A')}<br>
                <b>Preprocessing Engine:</b> {model_info.get('preprocessing_engine', 'N/A')}<br>
                <b>Selection Metric:</b> {model_info.get('selection_metric', 'N/A')}<br>
                <b>Best Metric Value:</b> {model_info.get('best_metric_value', 'N/A')}<br>
                <b>Timestamp:</b> {model_info.get('timestamp', 'N/A')}
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.error(f"Could not load model information: {str(e)}")
        
        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()