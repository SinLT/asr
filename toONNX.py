import transformers
from transformers import AutoTokenizer, Wav2Vec2ForCTC
from torchaudio.models.wav2vec2.utils import import_huggingface_model
import torch.onnx

original = Wav2Vec2ForCTC.from_pretrained("./jonatasgrosman-1665986652")
imported = import_huggingface_model(original)
imported.eval()

input_size = 100000
AUDIO_MAXLEN = input_size
dummy_input = torch.randn(1, input_size, requires_grad=True)

torch.onnx.export(imported,         # model being run
         dummy_input,       # model input (or a tuple for multiple inputs)
         "asr3.onnx",       # where to save the model
         export_params=True,  # store the trained parameter weights inside the model file
         opset_version=10,    # the ONNX version to export the model to
         do_constant_folding=True,  # whether to execute constant folding for optimization
         input_names = ['modelInput'],   # the model's input names
         output_names = ['modelOutput'], # the model's output names
         dynamic_axes={'modelInput' : {0 : 'batch_size'},    # variable length axes
                                'modelOutput' : {0 : 'batch_size'}})