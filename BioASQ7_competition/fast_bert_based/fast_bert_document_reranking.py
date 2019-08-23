
import torch, datetime, logging
from fast_bert.data_cls     import BertDataBunch
from fast_bert.learner_cls  import BertLearner
from fast_bert.metrics      import accuracy
from pathlib                import Path

torch.cuda.empty_cache()
run_start_time = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')

DATA_PATH   = Path('/home/dpappas/fast_bert_models/doc_rerank/data/')
LABEL_PATH  = Path('/home/dpappas/fast_bert_models/doc_rerank/labels/')
MODEL_PATH  = Path('/home/dpappas/fast_bert_models/doc_rerank/models/')
OUTPUT_PATH = Path('/home/dpappas/fast_bert_models/doc_rerank/models/output')
LOG_PATH    = Path('/home/dpappas/fast_bert_models/doc_rerank/logs/')
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
    train_file='train.csv',
    val_file='val.csv',
    label_file='labels.csv',
    text_col='text',
    label_col='label',
    batch_size_per_gpu=16,
    max_seq_length=512,
    multi_gpu=True,
    multi_label=False,
    model_type='bert')






