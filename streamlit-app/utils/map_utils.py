import streamlit as st

def get_sector_coordinates():
    """Get coordinates for Santiago sectors"""
    return {
        'las condes': {'lat': -33.4172, 'lng': -70.5476, 'zoom': 13},
        'lo barnechea': {'lat': -33.3500, 'lng': -70.5000, 'zoom': 13},
        'vitacura': {'lat': -33.3900, 'lng': -70.5700, 'zoom': 13},
        'nunoa': {'lat': -33.4569, 'lng': -70.5975, 'zoom': 13},
        'providencia': {'lat': -33.4372, 'lng': -70.6178, 'zoom': 13},
        'la reina': {'lat': -33.4450, 'lng': -70.5350, 'zoom': 13}
    }

def create_map(latitude, longitude, sector):
    """Create an interactive map with the property location"""
    
    # Basic HTML map using Leaflet
    map_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css" />
        <style>
            #map {{ height: 100%; width: 100%; }}
            body {{ margin: 0; padding: 0; }}
        </style>
    </head>
    <body>
        <div id="map"></div>
        <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
        <script>
            var map = L.map('map').setView([{latitude}, {longitude}], 13);
            
            L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
                attribution: '© OpenStreetMap contributors'
            }}).addTo(map);
            
            // Add marker for property location
            var marker = L.marker([{latitude}, {longitude}]).addTo(map);
            marker.bindPopup('<b>Property Location</b><br>Sector: {sector}<br>Lat: {latitude}<br>Lng: {longitude}').openPopup();
            
            // Add sector boundaries (approximate circles)
            var sectorCoords = {{
                'las condes': [-33.4172, -70.5476],
                'lo barnechea': [-33.3500, -70.5000],
                'vitacura': [-33.3900, -70.5700],
                'nunoa': [-33.4569, -70.5975],
                'providencia': [-33.4372, -70.6178],
                'la reina': [-33.4450, -70.5350]
            }};
            
            Object.keys(sectorCoords).forEach(function(sectorName) {{
                var coords = sectorCoords[sectorName];
                var circle = L.circle(coords, {{
                    color: sectorName === '{sector}' ? '#ff0000' : '#2563eb',
                    fillColor: sectorName === '{sector}' ? '#ff0000' : '#3b82f6',
                    fillOpacity: sectorName === '{sector}' ? 0.3 : 0.1,
                    radius: 3000
                }}).addTo(map);
                
                circle.bindPopup('<b>' + sectorName.charAt(0).toUpperCase() + sectorName.slice(1) + '</b>');
            }});
        </script>
    </body>
    </html>
    """
    
    return map_html

@st.cache_data
def get_santiago_bounds():
    """Get Santiago city bounds for validation"""
    return {
        'lat_min': -33.525,
        'lat_max': -33.305,
        'lng_min': -70.644,
        'lng_max': -70.432
    }

def is_valid_coordinate(lat, lng):
    """Check if coordinates are within Santiago bounds"""
    bounds = get_santiago_bounds()
    return (bounds['lat_min'] <= lat <= bounds['lat_max'] and 
            bounds['lng_min'] <= lng <= bounds['lng_max'])

def get_sector_info(sector):
    """Get detailed information about a sector"""
    sector_info = {
        'las condes': {
            'name': 'Las Condes',
            'description': 'Upscale residential and business district',
            'avg_price_range': '150,000 - 400,000 UF'
        },
        'vitacura': {
            'name': 'Vitacura',
            'description': 'Exclusive residential area with luxury properties',
            'avg_price_range': '200,000 - 500,000 UF'
        },
        'lo barnechea': {
            'name': 'Lo Barnechea',
            'description': 'Mountainous area with large houses and condos',
            'avg_price_range': '180,000 - 450,000 UF'
        },
        'nunoa': {
            'name': 'Ñuñoa',
            'description': 'Traditional residential neighborhood',
            'avg_price_range': '80,000 - 200,000 UF'
        },
        'providencia': {
            'name': 'Providencia',
            'description': 'Central urban area with apartments and offices',
            'avg_price_range': '100,000 - 250,000 UF'
        },
        'la reina': {
            'name': 'La Reina',
            'description': 'Residential area near the mountains',
            'avg_price_range': '120,000 - 300,000 UF'
        }
    }
    
    return sector_info.get(sector, {
        'name': sector.title(),
        'description': 'Santiago residential area',
        'avg_price_range': 'Variable'
    })