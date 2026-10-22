import os, glob, re

frontend_dir = r'c:\Users\NISHANTH S\Documents\Project\AI-Powered-Heavy-Rainfall-and-Flood-Early-Warning-System\AI-Powered-Heavy-Rainfall-and-Flood-Early-Warning-System\frontend'
html_files = glob.glob(os.path.join(frontend_dir, '*.html'))

for file in html_files:
    if 'login.html' in file or 'index.html' in file: continue
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Wrap the status pill and clock in a flex container that NEVER wraps
    replacement = '''<div style="display: flex; gap: 12px; align-items: center; white-space: nowrap; flex-wrap: nowrap; flex-shrink: 0;">
          <div id="backendStatusPill" class="status-pill-backend" title="API Status">
            <span class="status-pulse-dot"></span>
            <span id="backendStatusText">Backend Connected</span>
          </div>
          <div class="live-clock-badge" id="liveClock">--:--:-- IST</div>
        </div>'''
        
    pattern = r'<div id="backendStatusPill".*?<div class="live-clock-badge" id="liveClock">--:--:-- IST</div>'
    
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Update cache bust
    new_content = new_content.replace('css/style.css?v=2', 'css/style.css?v=3')
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(new_content)
print('Wrapped clock and pill in hardcoded flex container in all HTML files.')
