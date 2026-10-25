import sys
import os
from pathlib import Path
import json

# ensure project root is on sys.path so 'src' imports resolve
sys.path.insert(0, os.getcwd())

from src.news.ingest import ingest_ticker

persist_dir = '.cache/demo_status'
for t in ['AAPL', 'MSFT', 'GOOGL']:
    res = ingest_ticker(t, None, cache_dir='.cache/demo_articles', persist_dir=persist_dir, max_articles=2)
    print('ingested', t, res)

print('\nStatus file content:')
p = Path(persist_dir) / 'status.json'
print('status exists', p.exists())
if p.exists():
    print(json.dumps(json.load(open(p)), indent=2))
else:
    print('No status file found')
