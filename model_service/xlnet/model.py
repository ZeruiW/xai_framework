import io
import os
import json
import base64
import transformers, torch
import gc
import numpy as np
from torch.nn import functional as F


from flask import (
    Blueprint, request, jsonify, send_file
)

bp = Blueprint('xlnet', __name__, url_prefix='/xlnet')


basedir = os.path.abspath(os.path.dirname(__file__))
# print(basedir)

# model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f'on {device}')

model_path = os.path.join(basedir, 'full_model_new')


def load_model(model_path):
  from transformers import (
      WEIGHTS_NAME,
      AdamW,
      XLNetConfig,
      XLNetForSequenceClassification,
      XLNetTokenizer,
      get_linear_schedule_with_warmup
  )
  MODEL_CLASSES = {
      "xlnet": (XLNetConfig, XLNetForSequenceClassification, XLNetTokenizer),
  }
  model_type = 'xlnet'
  config_class, model_class, tokenizer_class = MODEL_CLASSES[model_type]
  model = model_class.from_pretrained(model_path).to(device)
  tokenizer = tokenizer_class.from_pretrained(model_path)
  return model, tokenizer

def clear_mem(model, tokenizer):
  torch.cuda.empty_cache()
  del model
  del tokenizer
  gc.collect()

model, tokenizer = load_model(model_path)

#inputs
def pipe_input_tensor(sample, label, tokenizer):
    max_length = len(sample)
    inputs = tokenizer.encode_plus(
        sample, None, add_special_tokens=True, max_length=max_length, truncation=True)
    inputs = {
        "input_ids": torch.tensor([inputs["input_ids"]]).to(device),
        "attention_mask": torch.tensor([inputs["attention_mask"]]).to(device),
        "labels": torch.tensor([label]).type(torch.LongTensor).to(device),
        "token_type_ids": torch.tensor([inputs["token_type_ids"]]).to(device)
    }
    return inputs


def convert_logits_to_prob(logits):
  # convert logit score to torch array
  torch_logits = torch.from_numpy(logits.cpu().detach().numpy())

  # get probabilities using softmax from logit score and convert it to numpy array
  probabilities_scores = F.softmax(torch_logits, dim=-1).numpy()
  return(probabilities_scores)

def get_label_prob(prob,label_list):
  prob_label = []
  count = 0
  for prob_sample in prob:
    prob_label.append(prob_sample[label_list[count]])
    count += 1
  return np.array(prob_label)


@bp.route('/', methods=['GET', 'POST'])
def pred():
    if request.method == 'POST':

        body = request.get_json()
        encodedsample = body['sample']
        sample = base64.b64decode(encodedsample).decode('utf-8')
        label = body['label']

        inputs = pipe_input_tensor(sample, label, tokenizer)
        print(inputs)
        # transform json to tensor

        # predict
        outputs = model(**inputs)
        logits = outputs.logits.cpu().detach().numpy()

        rs = {}
        rs['logits'] = logits.tolist()

        return jsonify(rs)

    elif request.method == 'GET':

        return 'xlnet'
