with open('frontend/css/style.css', encoding='utf-8') as f:
    for i, l in enumerate(f):
        if 'map' in l.lower() or 'leaflet' in l.lower():
            print(f'{i+1}: {l.strip()[:100]}')
