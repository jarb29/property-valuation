// Debug version with extensive logging
console.log('Debug script loaded');

document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded - starting debug initialization');
    
    // Check if elements exist
    const latInput = document.getElementById('latitude');
    const lngInput = document.getElementById('longitude');
    const mapElement = document.getElementById('map');
    
    console.log('Elements found:', {
        latitude: !!latInput,
        longitude: !!lngInput,
        map: !!mapElement
    });
    
    if (!latInput || !lngInput || !mapElement) {
        console.error('Missing required elements!');
        return;
    }
    
    // Initialize map
    const map = L.map('map').setView([-33.4372, -70.5476], 11);
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png').addTo(map);
    
    let marker;
    
    // Map click handler
    map.on('click', function(e) {
        const lat = parseFloat(e.latlng.lat.toFixed(6));
        const lng = parseFloat(e.latlng.lng.toFixed(6));
        
        console.log('Map clicked:', lat, lng);
        
        // Update inputs
        latInput.value = lat;
        lngInput.value = lng;
        
        console.log('Input values after update:', latInput.value, lngInput.value);
        
        // Update marker
        if (marker) {
            map.removeLayer(marker);
        }
        
        marker = L.marker([lat, lng]).addTo(map);
        marker.bindPopup(`Lat: ${lat}, Lng: ${lng}`).openPopup();
    });
    
    // Test form submission
    const form = document.getElementById('valuationForm');
    if (form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            console.log('Form submitted');
            
            const formData = new FormData(form);
            const data = Object.fromEntries(formData.entries());
            console.log('Form data:', data);
            
            // Test fetch
            fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ features: data })
            })
            .then(response => {
                console.log('Response status:', response.status);
                return response.json();
            })
            .then(result => {
                console.log('Prediction result:', result);
            })
            .catch(error => {
                console.error('Prediction error:', error);
            });
        });
    }
    
    console.log('Debug initialization complete');
});