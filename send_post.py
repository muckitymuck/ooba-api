import sys
import json
import requests

if not sys.argv[1]:
    print("No prompt sent. Use: send_post.py 'Tell me about Koalas.'")
    sys.exit()

#make api respect temp, etc
data = {
    "prompt": sys.argv[1],
    "temperature": 0.7,
}

r = requests.post("http://wintermute:7861/generate", data=json.dumps(data))
#r = requests.post("http://localhost:7861/generate", data=json.dumps(data))
print(r.status_code)
print(r.status_code==200)
if r.status_code==200:
    print(r.json())