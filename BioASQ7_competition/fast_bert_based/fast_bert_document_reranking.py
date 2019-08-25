
import sys, torch, datetime, logging
from torch                  import Tensor
from fast_bert.data_cls     import BertDataBunch
from fast_bert.learner_cls  import BertLearner
from fast_bert.metrics      import accuracy
from pathlib                import Path
from fast_bert.metrics      import accuracy_multilabel, accuracy, fbeta, roc_auc

torch.cuda.empty_cache()
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

device = torch.device('cuda')
if torch.cuda.device_count() > 1:
    multi_gpu = True
else:
    multi_gpu = False

DATA_PATH   = Path('/home/dpappas/fast_bert_models/doc_rerank/data/')
LABEL_PATH  = Path('/home/dpappas/fast_bert_models/doc_rerank/labels/')
MODEL_PATH  = Path('/home/dpappas/fast_bert_models/doc_rerank/models/')
OUTPUT_PATH = Path('/home/dpappas/fast_bert_models/doc_rerank/models/output')
LOG_PATH    = Path('/home/dpappas/fast_bert_models/doc_rerank/logs/')
DATA_PATH.mkdir(exist_ok=True)
LABEL_PATH.mkdir(exist_ok=True)
MODEL_PATH.mkdir(exist_ok=True)
LOG_PATH.mkdir(exist_ok=True)
OUTPUT_PATH.mkdir(exist_ok=True)

logfile = str(LOG_PATH/'log-{}-{}.txt'.format(run_start_time, 'doc_rerank_bioasq7'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger()

databunch = BertDataBunch(
    DATA_PATH,
    LABEL_PATH,
    tokenizer='bert-base-uncased',
    train_file='/home/dpappas/fast_bert_models/doc_rerank/train.csv',
    val_file='/home/dpappas/fast_bert_models/doc_rerank/val.csv',
    label_file='/home/dpappas/fast_bert_models/doc_rerank/labels.csv',
    text_col='text',
    label_col='label',
    batch_size_per_gpu=6,
    max_seq_length=512,
    multi_gpu=multi_gpu,
    multi_label=False,
    model_type='bert'
)

def roc_auc_2(y_pred: Tensor, y_true: Tensor):
    return (roc_auc(y_pred[:,1], y_true))

def fbeta_2(y_pred: Tensor, y_true: Tensor):
    return (fbeta(y_pred[:,1].squeeze().unsqueeze(-1), y_true.unsqueeze(-1)))

metrics = []
metrics.append({'name': 'accuracy', 'function': accuracy})
metrics.append({'name': 'roc_auc',  'function': roc_auc_2})
metrics.append({'name': 'fbeta',    'function': fbeta_2})

learner = BertLearner.from_pretrained_model(
    databunch,
    pretrained_path='bert-base-uncased',
    metrics=metrics,
    device=device,
    logger=logger,
    output_dir=OUTPUT_PATH,
    finetuned_wgts_path=None,
    warmup_steps=500,
    multi_gpu=multi_gpu,
    is_fp16=False,
    multi_label=False,
    logging_steps=-1
)

learner.fit(
    epochs=6,
	lr=6e-5,
    validate=True, 	# Evaluate the model after each epoch
    schedule_type="warmup_cosine",
	optimizer_type="lamb"
)

learner.save_and_reload(MODEL_PATH, "trained_model_name")

# CUDA_VISIBLE_DEVICES=1 python3.6 train_fast_bert_doc_rerank.py



