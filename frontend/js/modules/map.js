/**
 * SIH26071 – Frontend Risk Map Module
 * Initializes Leaflet maps and renders geospatial flood risk layers.
 */

export function initRiskMap(containerId, locations = []) {
    const mapElement = document.getElementById(containerId);
    if (!mapElement || typeof L === 'undefined') return null;

    const map = L.map(containerId).setView([20.5937, 78.9629], 5);

    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors',
        maxZoom: 18
    }).addTo(map);

    locations.forEach(loc => {
        if (loc.latitude && loc.longitude) {
            const circle = L.circleMarker([loc.latitude, loc.longitude], {
                radius: 6,
                fillColor: loc.risk_color || '#10b981',
                color: '#ffffff',
                weight: 1,
                opacity: 1,
                fillOpacity: 0.8
            }).addTo(map);

            circle.bindPopup(`
                <div style="font-family: sans-serif; font-size: 13px;">
                    <strong>Flood Risk: ${loc.flood_probability_pct || 0}%</strong><br>
                    <span>Risk Level: ${loc.risk_level || 'LOW'}</span><br>
                    <span>Rainfall: ${loc.rainfall || 0} mm</span><br>
                    <span>Water Level: ${loc.water_level || 0} m</span>
                </div>
            `);
        }
    });

    return map;
}
