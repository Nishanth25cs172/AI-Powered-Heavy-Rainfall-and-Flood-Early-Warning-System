/**
 * AI-Powered Heavy Rainfall and Flood Early-Warning System - Authentication & Portal Access Controller
 */

async function handleLogin(e) {
  if (e) e.preventDefault();
  const form = document.getElementById('loginForm');
  const msgEl = document.getElementById('loginMessage');
  const btn = document.getElementById('btnLoginSubmit');

  if (!form) return;

  const formData = new FormData(form);
  const payload = {
    username: formData.get('username') || 'admin',
    password: formData.get('password') || 'admin123'
  };

  if (btn) {
    btn.disabled = true;
    btn.innerHTML = 'Connecting to Command Center...';
  }
  if (msgEl) msgEl.textContent = '';

  try {
    const res = await DisasterAPI.login(payload.username, payload.password);
    if (res.success) {
      localStorage.setItem('sih_login', '1');
      if (msgEl) {
        msgEl.style.color = '#34d399';
        msgEl.textContent = '✓ Access granted. Initializing Command Center...';
      }
      setTimeout(() => {
        window.location.href = 'dashboard.html';
      }, 400);
    } else {
      if (msgEl) {
        msgEl.style.color = '#f87171';
        msgEl.textContent = res.message || 'Invalid credentials. Use demo: admin / admin123';
      }
    }
  } catch (err) {
    console.warn('Login request failed:', err);
    // Allow demo entry if backend is temporarily unreachable
    localStorage.setItem('sih_login', '1');
    window.location.href = 'dashboard.html';
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.innerHTML = 'Enter Command Center';
    }
  }
}

function handleDemoLaunch() {
  const form = document.getElementById('loginForm');
  if (form) {
    form.username.value = 'admin';
    form.password.value = 'admin123';
  }
  handleLogin();
}

function handleGuestEntry() {
  localStorage.setItem('sih_login', '1');
  window.location.href = 'dashboard.html';
}

window.addEventListener('DOMContentLoaded', () => {
  // Initialize canvas storm particles on the landing page
  const canvas = document.getElementById('heroStormCanvas');
  if (canvas && window.StormEngine) {
    window.heroStorm = new StormEngine('heroStormCanvas', {
      particleCount: 160,
      speed: 20,
      lightningEnabled: true
    });
  }

  // Check backend status for header pill
  DisasterAPI.ping().then(isOnline => {
    const statusPill = document.getElementById('heroBackendStatus');
    if (statusPill) {
      if (isOnline) {
        statusPill.innerHTML = '<span class="status-pulse-dot"></span> AI SYSTEM ONLINE · BACKEND CONNECTED';
      } else {
        statusPill.innerHTML = '<span class="status-pulse-dot" style="background:#f59e0b;box-shadow:0 0 10px #f59e0b;"></span> LOCAL MODE ACTIVE';
      }
    }
  });
});
