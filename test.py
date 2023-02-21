from huggingsound import SpeechRecognitionModel
from flask import Flask, request, jsonify
from waitress import serve
import torch

app = Flask(__name__)

device = torch.device('cpu')
model = SpeechRecognitionModel(model_path="./jonatasgrosman-1665986652", device=device)
# audio_paths = ["/mnt/d/AI-data/MedicalCorpus/process.wav", "/mnt/d/AI-data/MedicalCorpus/test.wav"]

@app.route('/asr', methods=["GET"])
def calculate():
    if request.method == 'GET':
        params = request.args
    url = params.get("url")
    audio_paths = []
    audio_paths.append(url)
    transcriptions = model.transcribe(audio_paths)
    txt = transcriptions[0]['transcription']
    print(txt)
    res = {"result": txt}
    return jsonify(content_type='application/json;charset=utf-8',
                   reason='success',
                   charset='utf-8',
                   status='200',
                   content=res)
 
serve(app, host="0.0.0.0", port=8868)