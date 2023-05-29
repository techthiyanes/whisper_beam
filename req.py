import requests
from scipy.io import wavfile
import json

endpoint = "https://apps.beam.cloud/qn7i6"
samplerate, data = wavfile.read('record.wav')
headers = {
    "Authorization": "Basic MWIzZDI3OWFkNDEyYTFmN2MxOWM1MDVmZTMyZDgyZTM6NTE5ZGZlMThkNTJiMzU1ZDI4NTRkMmYxMDA0YTM1NWE=",
    "Content-Type": "application/json",
    "Accept": "*/*",
    "Accept-Encoding": "gzip, deflate",
    "Connection": "keep-alive"
}
d = {
    "audio": {
        "rate": samplerate,
        "data": data.tolist(),
    }
}

r = requests.post(endpoint, headers=headers, data=json.dumps(d))
print(r.content)