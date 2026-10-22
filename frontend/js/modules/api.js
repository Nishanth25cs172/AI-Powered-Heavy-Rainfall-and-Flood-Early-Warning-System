/**
 * SIH26071 – Frontend API Module
 * Provides promise-based HTTP client wrappers for backend REST endpoints.
 */

export async function fetchHealth() {
    const res = await fetch('/api/health');
    return res.json();
}

export async function fetchDashboardData() {
    const res = await fetch('/api/dashboard');
    return res.json();
}

export async function predictRainfall(payload) {
    const res = await fetch('/api/predict-rainfall', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    return res.json();
}

export async function predictFlood(payload) {
    const res = await fetch('/api/predict-flood', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    return res.json();
}

export async function predictCompoundRisk(payload) {
    const res = await fetch('/api/predict-risk', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
    });
    return res.json();
}
