import onnx
import onnxruntime
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import scipy.signal as sps
import os
from pythainlp.util import normalize
from flask import Flask, request, jsonify
from waitress import serve
import re

with open("./jonatasgrosman-1665986652/vocab.json","r",encoding="utf-8-sig") as f:
  d = eval(f.read())

app = Flask(__name__)

input_size = 100000
new_rate = 16000
AUDIO_MAXLEN = input_size
ort_session = onnxruntime.InferenceSession('asr3.onnx') # load onnx model
res = dict((v,k) for k,v in d.items())

def _normalize(x): #
  """You must call this before padding.
  Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/src/wav2vec2/processor.py#L101
  Fork TF to numpy
  """
  # -> (1, seqlen)
  mean = np.mean(x, axis=-1, keepdims=True)
  var = np.var(x, axis=-1, keepdims=True)
  return np.squeeze((x - mean) / np.sqrt(var + 1e-5))
def remove_adjacent(item): # code from https://stackoverflow.com/a/3460423
  nums = list(item)
  a = nums[:1]
  for item in nums[1:]:
    if item != a[-1]:
      a.append(item)
  return ''.join(a)
def asr(path):
    """
    Code from https://github.com/vasudevgupta7/gsoc-wav2vec2/blob/main/notebooks/wav2vec2_onnx.ipynb
    Fork TF to numpy
    """
    sampling_rate, data = wavfile.read(path)
    samples = round(len(data) * float(new_rate) / sampling_rate)
    new_data = sps.resample(data, samples)
    speech = np.array(new_data, dtype=np.float32)
    speech = _normalize(speech)[None]
    padding = np.zeros((speech.shape[0], AUDIO_MAXLEN - speech.shape[1]))
    speech = np.concatenate([speech, padding], axis=-1).astype(np.float32)
    ort_inputs = {"modelInput": speech}
    ort_outs = ort_session.run(None, ort_inputs)
    prediction = np.argmax(ort_outs, axis=-1)
    # Text post processing
    _t1 = ''.join([res[i] for i in list(prediction[0][0])])
    return normalize(''.join([remove_adjacent(j) for j in _t1.split("[PAD]")]))

@app.route('/asr', methods=["GET"])
def calculate():
    if request.method == 'GET':
        params = request.args
    url = params.get("url")
    txt = asr(url)
    txt = re.sub("[<pad>|]", "", txt)
    print(txt)
    res = {"result": txt}
    return jsonify(content_type='application/json;charset=utf-8',
                   reason='success',
                   charset='utf-8',
                   status='200',
                   content=res)
 
serve(app, host="0.0.0.0", port=8868)