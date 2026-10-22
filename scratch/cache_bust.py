import os, glob

frontend_dir = r'c:\Users\NISHANTH S\Documents\Project\AI-Powered-Heavy-Rainfall-and-Flood-Early-Warning-System\AI-Powered-Heavy-Rainfall-and-Flood-Early-Warning-System\frontend'
html_files = glob.glob(os.path.join(frontend_dir, '*.html'))

for file in html_files:
    with open(file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Cache bust the CSS
    content = content.replace('href="css/style.css"', 'href="css/style.css?v=2"')
    
    with open(file, 'w', encoding='utf-8') as f:
        f.write(content)
print('Cache busting applied to CSS links.')
