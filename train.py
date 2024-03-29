from huggingsound import TrainingArguments, ModelArguments, SpeechRecognitionModel, TokenSet
import os, random, time
import torch
import numpy as np
import sys

# model_path = sys.argv[1]

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
# 设置随机数种子
setup_seed(42)

# model_path = "wav2vec2-large-chinese-zh-cn"
# model_path = "wav2vec2-large-xlsr-53-chinese-zh-cn"
# model_path = "ydshieh/wav2vec2-large-xlsr-53-chinese-zh-cn-gpt"
# model_path = "wbbbbb/wav2vec2-large-chinese-zh-cn"
# model_path = "jonatasgrosman/wav2vec2-large-xlsr-53-chinese-zh-cn"
model_path = "./jonatasgrosman-1665986652"
# model_path = "wav2vec2-large-xlsr-53-chinese-zn-cn-aishell1"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SpeechRecognitionModel(model_path=model_path, device=device)
now = int(time.time())
# output_dir = f'./checkpoint-{now}'
output_dir = f'./jonatasgrosman-{now}'

# first of all, you need to define your model's token set
# however, the token set is only needed for non-finetuned models
# if you pass a new token set for an already finetuned model, it'll be ignored during training
# tokens = ["A", "B", "C", "D", "F", "G", "H", "I", "M", "N", "P", "R", "S", "T"]
# token_set = TokenSet(tokens)

# define your train/eval data
train_data = []

def read_data(dataset_name):
    data_len = len(os.listdir(f'{dataset_name}/Split_TXT'))
    print(data_len)
    for cnt in range(1, data_len+1):
        transcription = open(f'{dataset_name}/Split_TXT/{cnt}.txt', encoding='utf-8').readline().strip()
        path = f'{dataset_name}/Split_WAV/{cnt}.wav'
        train_data.append(
            {"path": path, "transcription":transcription}
        )

def read_data2(dataset_name):
    data = os.listdir(f'{dataset_name}/Split_TXT')
    data_len = len(os.listdir(f'{dataset_name}/Split_TXT'))
    print(data_len)
    for cnt in range(1, data_len+1):
        (filename, extension) = os.path.splitext(data[cnt - 1])
        transcription = open(f'{dataset_name}/Split_TXT/{filename}.txt', encoding='utf-8').readline().strip()
        path = f'{dataset_name}/Split_WAV/{filename}.wav'
        train_data.append(
            {"path": path, "transcription":transcription}
        )
# read_data('Shanghai_Dialect_Conversational_Speech_Corpus')
# read_data('Shanghai_Dialect_Scripted_Speech_Corpus_Daily_Use_Sentence')
read_data2('/mnt/d/AI-data/MedicalCorpus/wav2vec2')
random.shuffle(train_data)
eval_ratio = 0.05
index = int(len(train_data) * eval_ratio)
eval_data = train_data[:index]
# train_data = train_data[index:]

# for debug
# eval_data = train_data[:10]
# train_data = train_data[10:20]


print('eval_data_len:', len(eval_data))
print('train_data_len:', len(train_data))

batch_size = 16
eval_steps = 100
# gradient_checkpointing=True,
# gradient_accumulation_steps=2,
# fp16=True,
training_args = TrainingArguments(
    save_steps=eval_steps,
    group_by_length=True,
    num_train_epochs=50,
    learning_rate=1e-4,
    eval_steps=eval_steps,
    weight_decay=0.005,
    save_total_limit=2,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    metric_for_best_model='cer'
)
model_args = ModelArguments(
    activation_dropout=0.1,
    hidden_dropout=0.1,
)

model.finetune(
    output_dir, 
    train_data=train_data, 
    eval_data=eval_data, # the eval_data is optional
    training_args=training_args,
    model_args=model_args
)
