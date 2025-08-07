import streamlit as st

def show_header():
    """Display the app header"""
    st.title("🏠 PropertyAI - Real Estate Valuation")
    st.markdown("""
    **Intelligent property valuation for Chilean real estate using advanced machine learning models.**
    
    Get instant, accurate property valuations based on location, size, and property characteristics.
    """)
    st.divider()

def show_footer():
    """Display the app footer"""
    st.divider()
    st.markdown("""
    ---
    **PropertyAI** | Powered by ONNX-optimized machine learning models
    
    💡 *This valuation is an estimate based on machine learning models trained on historical data. 
    Actual market value may vary based on current market conditions, property condition, and other factors.*
    """)

def format_currency(amount):
    """Format amount as Chilean Peso currency"""
    return f"${amount:,.0f} CLP"

def show_prediction_card(prediction, features, prediction_time):
    """Display prediction results in a card format"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(
            "Estimated Property Value",
            format_currency(prediction),
            delta=None
        )
        
        price_per_m2 = prediction / features['net_usable_area']
        st.metric(
            "Price per m²",
            format_currency(price_per_m2),
            delta=None
        )
    
    with col2:
        st.metric(
            "Prediction Time",
            f"{prediction_time*1000:.1f} ms",
            delta=None
        )
        
        st.metric(
            "Property Type",
            features['type'].title(),
            delta=None
        )

def show_property_summary(features):
    """Display property summary"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Type", features['type'].title())
        st.metric("Usable Area", f"{features['net_usable_area']} m²")
    
    with col2:
        st.metric("Sector", features['sector'].replace("_", " ").title())
        st.metric("Total Area", f"{features['net_area']} m²")
    
    with col3:
        st.metric("Rooms", f"{features['n_rooms']}")
        st.metric("Bathrooms", f"{features['n_bathroom']}")

def show_model_metrics(model_info):
    """Display model performance metrics"""
    metrics = model_info.get('metrics', {})
    
    if metrics:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if 'rmse' in metrics:
                st.metric("RMSE", f"{metrics['rmse']:.2f}")
        
        with col2:
            if 'mae' in metrics:
                st.metric("MAE", f"{metrics['mae']:.2f}")
        
        with col3:
            if 'mape' in metrics:
                st.metric("MAPE", f"{metrics['mape']:.4f}")

def validate_coordinates(lat, lng):
    """Validate if coordinates are within Santiago bounds"""
    return (-33.525 <= lat <= -33.305) and (-70.644 <= lng <= -70.432)

def show_coordinate_warning(lat, lng):
    """Show warning if coordinates are outside valid range"""
    if not validate_coordinates(lat, lng):
        st.warning("⚠️ Coordinates are outside the valid range for Santiago. Results may be less accurate.")
        return False
    return True