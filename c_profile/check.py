import json
from collections import Counter
data = json.load(open('results/schema_links_all_dbs_500per_20260306_163328.json'))
if isinstance(data, list):
    dbs = Counter(v['db_id'] for v in data)
else:
    dbs = Counter(v['db_id'] for v in data.values())
for db, count in sorted(dbs.items()):
    print(f'{db}: {count}')
print('Total:', sum(dbs.values()))
