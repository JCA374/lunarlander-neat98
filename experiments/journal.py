"""Add a journal note to the site log:  python experiments/journal.py "title" "text" """
import json, os, sys, time
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'journal.jsonl')
with open(path, 'a') as f:
    f.write(json.dumps({'t': time.strftime('%Y-%m-%d %H:%M:%S'), 'title': sys.argv[1], 'text': sys.argv[2]}) + '\n')
print('noted')
