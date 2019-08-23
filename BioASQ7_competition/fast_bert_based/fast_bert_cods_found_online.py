
from    pathlib                 import Path
from    fast_bert.data_cls      import BertDataBunch
from    fast_bert.learner_cls   import BertLearner
from    fast_bert.metrics       import accuracy
from    box                     import Box
from    tqdm                    import tqdm, trange
from    fast_bert.metrics       import accuracy_multilabel, accuracy_thresh, fbeta, roc_auc
import  logging
import  torch
import  sys
import  logging
import  datetime

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

BERT_PRETRAINED_PATH    = Path('../../bert_models/pretrained-weights/uncased_L-12_H-768_A-12/')
FINETUNED_PATH          = None

args = Box({
    "run_text"                      : "multilabel toxic comments with freezable layers",
    "train_size"                    : -1,
    "val_size"                      : -1,
    "log_path"                      : LOG_PATH,
    "full_data_dir"                 : DATA_PATH,
    "data_dir"                      : DATA_PATH,
    "task_name"                     : "toxic_classification_lib",
    "no_cuda"                       : False,
    "bert_model"                    : BERT_PRETRAINED_PATH,
    "output_dir"                    : OUTPUT_PATH,
    "max_seq_length"                : 512,
    "do_train"                      : True,
    "do_eval"                       : True,
    "do_lower_case"                 : True,
    "train_batch_size"              : 8,
    "eval_batch_size"               : 16,
    "learning_rate"                 : 5e-5,
    "num_train_epochs"              : 6,
    "warmup_proportion"             : 0.0,
    "local_rank"                    : -1,
    "gradient_accumulation_steps"   : 1,
    "optimize_on_cpu"               : False,
    "fp16"                          : True,
    "fp16_opt_level"                : "O1",
    "weight_decay"                  : 0.0,
    "adam_epsilon"                  : 1e-8,
    "max_grad_norm"                 : 1.0,
    "max_steps"                     : -1,
    "warmup_steps"                  : 500,
    "logging_steps"                 : 50,
    "eval_all_checkpoints"          : True,
    "overwrite_output_dir"          : True,
    "overwrite_cache"               : False,
    "seed"                          : 42,
    "loss_scale"                    : 128,
    "model_name"                    : 'xlnet-base-cased',
    "model_type"                    : 'xlnet'
})

device = torch.device('cuda')
if torch.cuda.device_count() > 1:
    args.multi_gpu = True
else:
    args.multi_gpu = False

logfile = str(LOG_PATH/'log-{}-{}.txt'.format(run_start_time, args["run_text"]))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.FileHandler(logfile), logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger()
logger.info(args)

databunch = BertDataBunch(
    args['data_dir'],
    LABEL_PATH,
    args.model_name,
    train_file='train.csv',
    val_file='val.csv',
    test_data='test.csv',
    text_col="comment_text",
    label_col=label_cols,
    batch_size_per_gpu=args['train_batch_size'],
    max_seq_length=args['max_seq_length'],
    multi_gpu=args.multi_gpu,
    multi_label=True,
    model_type=args.model_type
)


print(databunch.train_dl.dataset[0][3])
num_labels = len(databunch.labels)
print(num_labels)

metrics = []
metrics.append({'name': 'accuracy_thresh', 'function': accuracy_thresh})
metrics.append({'name': 'roc_auc', 'function': roc_auc})
metrics.append({'name': 'fbeta', 'function': fbeta})

learner = BertLearner.from_pretrained_model(
    databunch,
    args.model_name,
    metrics=metrics,
    device=device,
    logger=logger,
    output_dir=args.output_dir,
    finetuned_wgts_path=FINETUNED_PATH,
    warmup_steps=args.warmup_steps,
    multi_gpu=args.multi_gpu,
    is_fp16=args.fp16,
    multi_label=True,
    logging_steps=0
)

learner.validate()
learner.save_model()


