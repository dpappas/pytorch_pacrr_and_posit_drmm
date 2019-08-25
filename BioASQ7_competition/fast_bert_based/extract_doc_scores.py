from fast_bert.prediction   import BertClassificationPredictor
from pathlib                import Path


DATA_PATH   = Path('/home/dpappas/fast_bert_models/doc_rerank/data/')
LABEL_PATH  = Path('/home/dpappas/fast_bert_models/doc_rerank/labels/')
MODEL_PATH  = Path('/home/dpappas/fast_bert_models/doc_rerank/models/')
LOG_PATH    = Path('/home/dpappas/fast_bert_models/doc_rerank/logs/')

# location for the pretrained BERT models
BERT_PRETRAINED_PATH = Path('../../bert_models/pretrained-weights/uncased_L-12_H-768_A-12/')


predictor = BertClassificationPredictor(
    model_path      = MODEL_PATH,
    pretrained_path = BERT_PRETRAINED_PATH,
    label_path      = LABEL_PATH,
    multi_label     = False
)

# Single prediction
single_prediction = predictor.predict("just get me result for this text")

# Batch predictions
texts = [
  "this is the first text",
  "this is the second text"
]

multiple_predictions = predictor.predict(texts)
