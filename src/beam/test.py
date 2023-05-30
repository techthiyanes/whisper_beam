from beam.utils.mock import MockAPI
from scipy.io import wavfile
import json

# Import your Beam app
from app import app

# Load your app into the MockAPI
mocker = MockAPI(app)

samplerate, data = wavfile.read('record.wav')
d = {
    "rate": samplerate,
    "data": data.tolist()
}

# Call the API
mocker.call(audio=json.dumps(d))
