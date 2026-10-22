import os, glob, re

frontend_dir = r'c:\Users\NISHANTH S\Documents\Project\AI-Powered-Heavy-Rainfall-and-Flood-Early-Warning-System\AI-Powered-Heavy-Rainfall-and-Flood-Early-Warning-System\frontend'
dashboard_path = os.path.join(frontend_dir, 'dashboard.html')

with open(dashboard_path, 'r', encoding='utf-8') as f:
    dash_content = f.read()

sidebar_header_match = re.search(r'<div class="sidebar-header">.*?</div>\s*(?=<nav)', dash_content, re.DOTALL)
app_header_match = re.search(r'<header class="app-header">.*?</header>', dash_content, re.DOTALL)

if not sidebar_header_match or not app_header_match:
    print('Could not find headers in dashboard.html')
    exit(1)

dash_sidebar_header = sidebar_header_match.group(0)
dash_app_header = app_header_match.group(0)

html_files = glob.glob(os.path.join(frontend_dir, '*.html'))
for file in html_files:
    if 'dashboard.html' in file or '_nav.html' in file or 'login.html' in file or 'index.html' in file:
        continue
    
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Replace sidebar header
    content = re.sub(r'<div class="sidebar-header">.*?</div>\s*(?=<nav)', dash_sidebar_header, content, flags=re.DOTALL)
    
    # Replace app header
    content = re.sub(r'<header class="app-header">.*?</header>', dash_app_header, content, flags=re.DOTALL)
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
print('Done replacing headers in all HTML files.')
