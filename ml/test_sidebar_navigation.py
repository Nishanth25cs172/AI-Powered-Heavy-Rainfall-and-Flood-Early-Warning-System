import urllib.request

base = "http://127.0.0.1:5000"
endpoints = [
    ("/dashboard.html", "Dashboard Overview"),
    ("/prediction.html", "Flood Prediction"),
    ("/alerts.html", "Redirecting to Early Warning Alerts"),
    ("/analytics.html", "Redirecting to Model Analytics"),
    ("/risk-map.html", "Redirecting to Geospatial Risk Map"),
    ("/results.html", "Redirecting to Prediction Results")
]

print("=== TESTING NAVIGATION ROUTING AND PAGE CONTENT ===")
for path, expected_phrase in endpoints:
    url = f"{base}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as resp:
        body = resp.read().decode('utf-8')
        has_phrase = expected_phrase in body
        print(f"[{'PASS' if has_phrase else 'FAIL'}] {path:18} -> Status {resp.status} | Phrase '{expected_phrase}': {has_phrase}")

# Check that dashboard.html contains the new left sidebar structure
with urllib.request.urlopen(f"{base}/dashboard.html") as resp:
    dash_html = resp.read().decode('utf-8')
    assert "class=\"app-sidebar\"" in dash_html, "Missing app-sidebar"
    assert "class=\"sidebar-header\"" in dash_html, "Missing sidebar-header"
    assert "AI-POWERED" in dash_html and "HEAVY RAINFALL" in dash_html, "Missing sidebar logo elements"
    assert "data-view=\"overview\"" in dash_html, "Missing dashboard nav item"
    assert "data-view=\"prediction\"" in dash_html, "Missing prediction nav item"
    assert "data-view=\"monitoring\"" in dash_html, "Missing monitoring nav item"
    assert "data-view=\"alerts\"" in dash_html, "Missing alerts nav item"
    assert "data-view=\"analytics\"" in dash_html, "Missing analytics nav item"
    assert "data-view=\"map\"" in dash_html, "Missing map nav item"
    assert "data-view=\"emergency\"" in dash_html, "Missing emergency nav item"
    assert "data-view=\"about\"" in dash_html, "Missing about nav item"
    assert "class=\"main-wrapper\"" in dash_html, "Missing main-wrapper"
    assert "class=\"main-header\"" in dash_html, "Missing main-header"
    assert "id=\"headerCurrentTitle\"" in dash_html, "Missing headerCurrentTitle"
    assert "id=\"btnSidebarToggle\"" in dash_html, "Missing btnSidebarToggle"
    print("\n[PASS] dashboard.html Left Sidebar and Main Wrapper structure verified 100%!")

# Check that style.css contains the left sidebar CSS rules
with urllib.request.urlopen(f"{base}/css/style.css") as resp:
    css_content = resp.read().decode('utf-8')
    assert ".app-sidebar {" in css_content, "Missing .app-sidebar in style.css"
    assert ".main-wrapper {" in css_content, "Missing .main-wrapper in style.css"
    assert ".main-header {" in css_content, "Missing .main-header in style.css"
    assert "width: 260px;" in css_content, "Missing 260px width in style.css"
    assert "position: fixed;" in css_content, "Missing position: fixed in style.css"
    print("[PASS] style.css Fixed Left Sidebar and Responsive Layout rules verified 100%!")
