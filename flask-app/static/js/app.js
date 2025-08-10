// Global variables
let map;
let marker;
let isMapInitialized = false;

// Santiago sectors with their approximate coordinates
const sectorCoordinates = {
    'las condes': { lat: -33.4172, lng: -70.5476, zoom: 13 },
    'lo barnechea': { lat: -33.3500, lng: -70.5000, zoom: 13 },
    'vitacura': { lat: -33.3900, lng: -70.5700, zoom: 13 },
    'nunoa': { lat: -33.4569, lng: -70.5975, zoom: 13 },
    'providencia': { lat: -33.4372, lng: -70.6178, zoom: 13 },
    'la reina': { lat: -33.4450, lng: -70.5350, zoom: 13 }
};

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    console.log('DOM loaded, initializing app...');
    
    // Add small delay to ensure all elements are rendered
    setTimeout(() => {
        initializeMap();
        initializeForm();
        initializeEventListeners();
    }, 100);
});

// Initialize Leaflet map
function initializeMap() {
    try {
        console.log('Initializing map...');
        
        const mapElement = document.getElementById('map');
        if (!mapElement) {
            console.error('Map element not found!');
            return;
        }
        
        // Santiago center coordinates
        const santiagoCenter = [-33.4372, -70.5476];
        
        map = L.map('map').setView(santiagoCenter, 11);
        
        // Add OpenStreetMap tiles
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: '© OpenStreetMap contributors',
            maxZoom: 18
        }).addTo(map);
        
        // Add click event to map
        map.on('click', function(e) {
            const lat = parseFloat(e.latlng.lat.toFixed(6));
            const lng = parseFloat(e.latlng.lng.toFixed(6));
            
            console.log('Map clicked:', lat, lng);
            
            // Update form inputs
            const latInput = document.getElementById('latitude');
            const lngInput = document.getElementById('longitude');
            
            if (latInput && lngInput) {
                latInput.value = lat;
                lngInput.value = lng;
                console.log('Updated inputs:', latInput.value, lngInput.value);
            }
            
            // Update marker
            updateMarker(lat, lng);
            
            // Validate coordinates
            validateCoordinates();
        });
        
        // Add sector boundaries (approximate)
        addSectorBoundaries();
        
        isMapInitialized = true;
        console.log('Map initialized successfully');
        
    } catch (error) {
        console.error('Error initializing map:', error);
        showAlert('Map initialization failed. Please refresh the page.', 'error');
    }
}

// Add sector boundaries to map
function addSectorBoundaries() {
    Object.entries(sectorCoordinates).forEach(([sector, coords]) => {
        const circle = L.circle([coords.lat, coords.lng], {
            color: '#2563eb',
            fillColor: '#3b82f6',
            fillOpacity: 0.1,
            radius: 3000
        }).addTo(map);
        
        circle.bindPopup(`<strong>${sector.charAt(0).toUpperCase() + sector.slice(1)}</strong>`);
    });
}

// Update marker on map
function updateMarker(lat, lng) {
    console.log('Updating marker:', lat, lng);
    
    if (marker) {
        map.removeLayer(marker);
    }
    
    marker = L.marker([lat, lng], {
        draggable: true
    }).addTo(map);
    
    marker.bindPopup(`
        <div style="text-align: center;">
            <strong>Selected Location</strong><br>
            Lat: ${lat}<br>
            Lng: ${lng}
        </div>
    `).openPopup();
    
    // Add drag event to marker
    marker.on('dragend', function(e) {
        const newLat = parseFloat(e.target.getLatLng().lat.toFixed(6));
        const newLng = parseFloat(e.target.getLatLng().lng.toFixed(6));
        
        console.log('Marker dragged to:', newLat, newLng);
        
        const latInput = document.getElementById('latitude');
        const lngInput = document.getElementById('longitude');
        
        if (latInput && lngInput) {
            latInput.value = newLat;
            lngInput.value = newLng;
        }
        
        validateCoordinates();
    });
}

// Initialize form functionality
function initializeForm() {
    const form = document.getElementById('valuationForm');
    const sectorSelect = document.getElementById('sector');
    
    // Handle sector change
    sectorSelect.addEventListener('change', function() {
        const selectedSector = this.value;
        console.log('Sector changed to:', selectedSector);
        
        if (selectedSector && sectorCoordinates[selectedSector]) {
            const coords = sectorCoordinates[selectedSector];
            console.log('Setting coordinates for sector:', coords);
            
            // Update map view
            if (isMapInitialized) {
                map.setView([coords.lat, coords.lng], coords.zoom);
            }
            
            // Set default coordinates
            const latInput = document.getElementById('latitude');
            const lngInput = document.getElementById('longitude');
            
            if (latInput && lngInput) {
                latInput.value = coords.lat;
                lngInput.value = coords.lng;
                console.log('Set sector coordinates:', latInput.value, lngInput.value);
            }
            
            // Update marker
            updateMarker(coords.lat, coords.lng);
        }
    });
    
    // Handle form submission
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        submitValuation();
    });
    
    // Real-time validation
    const inputs = form.querySelectorAll('input, select');
    inputs.forEach(input => {
        input.addEventListener('input', validateForm);
        input.addEventListener('blur', validateForm);
    });
}

// Initialize event listeners
function initializeEventListeners() {
    // Modal close events
    const modal = document.getElementById('resultsModal');
    const closeBtn = document.querySelector('.modal-close');
    
    closeBtn.addEventListener('click', closeModal);
    
    window.addEventListener('click', function(e) {
        if (e.target === modal) {
            closeModal();
        }
    });
    
    // Initialize draggable functionality
    makeDraggable(modal);
    
    // Alert close events
    document.addEventListener('click', function(e) {
        if (e.target.classList.contains('alert-close')) {
            e.target.parentElement.remove();
        }
    });
    
    // Coordinate input validation
    const latInput = document.getElementById('latitude');
    const lngInput = document.getElementById('longitude');
    
    [latInput, lngInput].forEach(input => {
        input.addEventListener('input', function() {
            validateCoordinates();
            
            const lat = parseFloat(latInput.value);
            const lng = parseFloat(lngInput.value);
            
            if (!isNaN(lat) && !isNaN(lng) && isValidCoordinate(lat, lng)) {
                updateMarker(lat, lng);
                if (isMapInitialized) {
                    map.setView([lat, lng], 15);
                }
            }
        });
    });
}

// Validate coordinates
function validateCoordinates() {
    const latInput = document.getElementById('latitude');
    const lngInput = document.getElementById('longitude');
    
    const lat = parseFloat(latInput.value);
    const lng = parseFloat(lngInput.value);
    
    if (!isNaN(lat) && !isNaN(lng)) {
        if (isValidCoordinate(lat, lng)) {
            latInput.classList.remove('error');
            lngInput.classList.remove('error');
            return true;
        } else {
            latInput.classList.add('error');
            lngInput.classList.add('error');
            showAlert('Coordinates are outside the valid range for Santiago', 'error');
            return false;
        }
    }
    return false;
}

// Check if coordinates are valid for Santiago
function isValidCoordinate(lat, lng) {
    return lat >= -33.525 && lat <= -33.305 && lng >= -70.644 && lng <= -70.432;
}

// Validate entire form
function validateForm() {
    const form = document.getElementById('valuationForm');
    const submitBtn = document.getElementById('submitBtn');
    
    // Get all required fields
    const fields = {
        type: document.getElementById('type'),
        sector: document.getElementById('sector'),
        net_usable_area: document.getElementById('net_usable_area'),
        net_area: document.getElementById('net_area'),
        n_rooms: document.getElementById('n_rooms'),
        n_bathroom: document.getElementById('n_bathroom'),
        latitude: document.getElementById('latitude'),
        longitude: document.getElementById('longitude')
    };
    
    let allFieldsFilled = true;
    
    // Validate each field and add highlighting
    Object.entries(fields).forEach(([fieldName, field]) => {
        const value = field.value.trim();
        const formGroup = field.closest('.form-group');
        
        if (!value) {
            field.classList.add('field-error');
            formGroup.classList.add('field-error');
            allFieldsFilled = false;
        } else {
            field.classList.remove('field-error');
            formGroup.classList.remove('field-error');
        }
    });
    
    // Check coordinates validity
    const coordsValid = validateCoordinates();
    
    const isValid = allFieldsFilled && coordsValid;
    
    console.log('Form validation:', {
        allFieldsFilled, coordsValid, isValid
    });
    
    submitBtn.disabled = !isValid;
    
    return isValid;
}

// Submit valuation request
async function submitValuation() {
    const form = document.getElementById('valuationForm');
    const submitBtn = document.getElementById('submitBtn');
    const spinner = submitBtn.querySelector('.loading-spinner');
    const btnText = submitBtn.querySelector('span');
    
    // Force validation to highlight missing fields
    if (!validateForm()) {
        showAlert('Please fill in all required fields correctly (highlighted in red)', 'error');
        
        // Scroll to first error field
        const firstError = document.querySelector('.field-error');
        if (firstError) {
            firstError.scrollIntoView({ behavior: 'smooth', block: 'center' });
            firstError.focus();
        }
        return;
    }
    
    // Show loading state
    submitBtn.disabled = true;
    spinner.style.display = 'block';
    btnText.textContent = 'Calculating...';
    
    try {
        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());
        
        // Convert numeric fields
        data.net_usable_area = parseFloat(data.net_usable_area);
        data.net_area = parseFloat(data.net_area);
        data.n_rooms = parseInt(data.n_rooms);
        data.n_bathroom = parseInt(data.n_bathroom);
        data.latitude = parseFloat(data.latitude);
        data.longitude = parseFloat(data.longitude);
        
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ features: data })
        });
        
        const result = await response.json();
        
        if (response.ok) {
            // Clear any previous validation errors
            clearValidationErrors();
            showResults(result, data);
        } else {
            // Handle validation errors from server
            if (result.validation_errors) {
                highlightErrorFields(result.validation_errors.error_fields);
                showAlert(result.error + ' (highlighted in red)', 'error');
            } else {
                throw new Error(result.error || 'Prediction failed');
            }
        }
        
    } catch (error) {
        console.error('Error:', error);
        showAlert(`Error: ${error.message}`, 'error');
    } finally {
        // Reset button state
        submitBtn.disabled = false;
        spinner.style.display = 'none';
        btnText.textContent = 'Calculate Property Value';
    }
}

// Show results in modal
function showResults(result, inputData) {
    const modal = document.getElementById('resultsModal');
    const content = document.getElementById('resultsContent');
    const modalContent = modal.querySelector('.modal-content');
    
    // Reset modal position
    modalContent.style.transform = 'translate(-50%, -50%)';
    
    const formattedPrice = `${result.prediction.toLocaleString('es-CL')} UF`;
    
    const pricePerM2 = Math.round(result.prediction / inputData.net_usable_area);
    const formattedPricePerM2 = `${pricePerM2.toLocaleString('es-CL')} UF/m²`;
    
    content.innerHTML = `
        <div class="results-container">
            <div class="result-main">
                <div class="price-display">
                    <h2>${formattedPrice}</h2>
                    <p>Estimated Property Value</p>
                </div>
                
                <div class="price-breakdown">
                    <div class="breakdown-item">
                        <span class="label">Price per m²:</span>
                        <span class="value">${formattedPricePerM2}</span>
                    </div>
                    <div class="breakdown-item">
                        <span class="label">Prediction Time:</span>
                        <span class="value">${(result.prediction_time * 1000).toFixed(1)}ms</span>
                    </div>
                    <div class="breakdown-item">
                        <span class="label">Model Version:</span>
                        <span class="value">${result.model_version}</span>
                    </div>
                </div>
            </div>
            
            <div class="property-summary">
                <h4><i class="fas fa-home"></i> Property Summary</h4>
                <div class="summary-grid">
                    <div class="summary-item">
                        <span class="label">Type:</span>
                        <span class="value">${inputData.type.charAt(0).toUpperCase() + inputData.type.slice(1)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Sector:</span>
                        <span class="value">${inputData.sector.charAt(0).toUpperCase() + inputData.sector.slice(1)}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Usable Area:</span>
                        <span class="value">${inputData.net_usable_area} m²</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Total Area:</span>
                        <span class="value">${inputData.net_area} m²</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Rooms:</span>
                        <span class="value">${inputData.n_rooms}</span>
                    </div>
                    <div class="summary-item">
                        <span class="label">Bathrooms:</span>
                        <span class="value">${inputData.n_bathroom}</span>
                    </div>
                </div>
            </div>
            
            <div class="disclaimer">
                <i class="fas fa-info-circle"></i>
                <p>This valuation is an estimate based on machine learning models trained on historical data. 
                Actual market value may vary based on current market conditions, property condition, and other factors.</p>
            </div>
        </div>
    `;
    
    modal.style.display = 'block';
    
    // Make modal draggable
    makeDraggable(modal);
    
    // Add resize listener for responsiveness
    window.addEventListener('resize', function() {
        modalContent.style.transform = 'translate(-50%, -50%)';
    });
}

// Close modal
function closeModal() {
    const modal = document.getElementById('resultsModal');
    modal.style.display = 'none';
}

// Make modal draggable
function makeDraggable(modal) {
    const header = modal.querySelector('.modal-header');
    const content = modal.querySelector('.modal-content');
    
    let isDragging = false;
    let currentX;
    let currentY;
    let initialX;
    let initialY;
    let xOffset = 0;
    let yOffset = 0;
    
    header.addEventListener('mousedown', dragStart);
    document.addEventListener('mousemove', drag);
    document.addEventListener('mouseup', dragEnd);
    
    function dragStart(e) {
        initialX = e.clientX - xOffset;
        initialY = e.clientY - yOffset;
        
        if (e.target === header || header.contains(e.target)) {
            isDragging = true;
            header.style.cursor = 'grabbing';
        }
    }
    
    function drag(e) {
        if (isDragging) {
            e.preventDefault();
            
            currentX = e.clientX - initialX;
            currentY = e.clientY - initialY;
            
            xOffset = currentX;
            yOffset = currentY;
            
            content.style.transform = `translate(calc(-50% + ${currentX}px), calc(-50% + ${currentY}px))`;
        }
    }
    
    function dragEnd() {
        isDragging = false;
        header.style.cursor = 'move';
    }
}

// Show alert message
function showAlert(message, type = 'info') {
    const alertsContainer = document.querySelector('.alerts') || createAlertsContainer();
    
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.innerHTML = `
        <i class="fas fa-${type === 'error' ? 'exclamation-triangle' : 'check-circle'}"></i>
        <span>${message}</span>
        <button class="alert-close">&times;</button>
    `;
    
    alertsContainer.appendChild(alert);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (alert.parentElement) {
            alert.remove();
        }
    }, 5000);
}

// Create alerts container if it doesn't exist
function createAlertsContainer() {
    const container = document.createElement('div');
    container.className = 'alerts';
    document.querySelector('.main').insertBefore(container, document.querySelector('.main').firstChild);
    return container;
}

// Highlight error fields based on server validation
function highlightErrorFields(errorFields) {
    console.log('Highlighting error fields:', errorFields);
    
    // Clear previous errors first
    clearValidationErrors();
    
    // Highlight each error field
    errorFields.forEach(fieldName => {
        const field = document.getElementById(fieldName);
        if (field) {
            const formGroup = field.closest('.form-group');
            field.classList.add('field-error');
            if (formGroup) {
                formGroup.classList.add('field-error');
            }
        }
    });
    
    // Scroll to first error field
    if (errorFields.length > 0) {
        const firstErrorField = document.getElementById(errorFields[0]);
        if (firstErrorField) {
            firstErrorField.scrollIntoView({ behavior: 'smooth', block: 'center' });
            setTimeout(() => firstErrorField.focus(), 500);
        }
    }
}

// Clear all validation error highlighting
function clearValidationErrors() {
    const errorFields = document.querySelectorAll('.field-error');
    errorFields.forEach(element => {
        element.classList.remove('field-error');
    });
}