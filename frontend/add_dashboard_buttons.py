with open(r'C:\Users\Mithilesh\OneDrive\Project\AIML\AI-Powered-Heavy-Rainfall-and-Flood-Early-Warning-System\frontend\dashboard.html', 'r', encoding='utf-8') as f:
    d = f.read()

target = '<section id="view-overview" class="view-section active">'
replacement = '''<section id="view-overview" class="view-section active">
        <div class="component-card" style="padding: 16px 20px; display: flex; justify-content: space-between; align-items: center;">
          <div style="font-weight: 700; font-size: 14px;">⚡ Quick Actions &amp; Intelligence</div>
          <div style="display: flex; gap: 12px;">
            <a href="analytics.html" class="btn-secondary">[View Analytics]</a>
            <a href="rainfall-prediction.html" class="btn-primary">[View Predictions]</a>
          </div>
        </div>'''

if target in d:
    d = d.replace(target, replacement)

with open(r'C:\Users\Mithilesh\OneDrive\Project\AIML\AI-Powered-Heavy-Rainfall-and-Flood-Early-Warning-System\frontend\dashboard.html', 'w', encoding='utf-8') as f:
    f.write(d)

print('dashboard.html updated with View Analytics and View Predictions buttons!')
