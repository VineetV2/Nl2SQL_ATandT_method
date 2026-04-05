import json
import sqlite3
import sys
from pathlib import Path

MINIDEV_ROOT = Path("/Users/vora/Documents/PyCharm/DAIL-SQL/c_profile/MINIDEV /dev_databases")

def execute_sql(sql, db_path):
    if not sql or not sql.strip(): return None
    try:
        conn = sqlite3.connect(str(db_path), timeout=5)
        def progress(): return 1
        conn.set_progress_handler(progress, 1000000)
        cursor = conn.execute(sql)
        rows = frozenset(cursor.fetchall())
        conn.close()
        return rows
    except Exception as e:
        return None

try:
    with open('/Users/vora/Documents/PyCharm/DAIL-SQL/c_profile/results/candidates_500q_openrouter.json') as f:
        data = json.load(f)
except Exception as e:
    print(f"Error reading JSON: {e}")
    exit(1)

results = data.get("results", [])
total = len(results)
if total == 0:
    print("No results found yet.")
    exit(0)

print(f"Evaluating {total} completed queries so far...\n")
sys.stdout.flush()

voted_correct = 0
not_voted_correct = 0

for i, r in enumerate(results, 1):
    db_id = r.get("db_id", "")
    db_path = MINIDEV_ROOT / db_id / f"{db_id}.sqlite"
    gold_sql = r.get("gold_sql", "")
    voted_sql = r.get("voted_sql", "")
    
    gold_result = execute_sql(gold_sql, db_path)
    
    if gold_result is not None:
        voted_result = execute_sql(voted_sql, db_path)
        if voted_result is not None and voted_result == gold_result:
            voted_correct += 1
            
        any_correct = False
        for cand in r.get("candidates", []):
            cand_sql = cand.get("sql", "")
            cand_result = execute_sql(cand_sql, db_path)
            if cand_result is not None and cand_result == gold_result:
                any_correct = True
                break
        if any_correct:
            not_voted_correct += 1
            
print("\n--- Final Results ---")
print(f"Total processed: {total}")
print(f"Oracle Accuracy (Any candidate correct): {not_voted_correct}/{total} = {not_voted_correct/total:.2%}")
print(f"Voted Accuracy (Final selection correct): {voted_correct}/{total} = {voted_correct/total:.2%}")
