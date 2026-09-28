with open('frontend/js/app.js', encoding='utf-8') as f:
    for i, l in enumerate(f):
        if 'leaflet' in l.lower() or 'initmap' in l.lower() or 'rendermappoints' in l.lower():
            print(f'{i+1}: {l.strip()[:100]}')
