from pathlib import Path
import json
from datetime import datetime

LOG_DIR = Path('logs')
now = datetime.utcnow().isoformat(timespec='microseconds') + 'Z'
updated = []
for p in LOG_DIR.glob('progress_*.json'):
    try:
        data = json.loads(p.read_text(encoding='utf-8'))
        data['last_updated'] = now
        p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        updated.append(str(p))
    except Exception as e:
        print(f'Failed to update {p}: {e}')

print('Updated files:')
for u in updated:
    print(u)
print('New timestamp:', now)
