


# download the files
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ccTfSBNofHIBCsZpHR2YU_7l86UMR460' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ccTfSBNofHIBCsZpHR2YU_7l86UMR460" -O weights.hdf5 && rm -rf /tmp/cookies.txt
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1W8ZXsMNxdq4s-KdgoL42ivSG49VOKRze' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1W8ZXsMNxdq4s-KdgoL42ivSG49VOKRze" -O options.json && rm -rf /tmp/cookies.txt

from pprint import pprint
from allennlp.modules.elmo import Elmo, batch_to_ids
# options_file    = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
# weight_file     = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
# options_file    = "/home/dpappas/bioasq_all/elmo_weights/options.json"
# weight_file     = "/home/dpappas/bioasq_all/elmo_weights/weights.hdf5"
options_file    = "/home/dpappas/for_ryan/elmo_weights/options.json"
weight_file     = "/home/dpappas/for_ryan/elmo_weights/weights.hdf5"
elmo            = Elmo(options_file, weight_file, 1, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences       = [['First', 'sentence', '.'], ['Another', '.']]
character_ids   = batch_to_ids(sentences)
pprint(character_ids.size())
embeddings      = elmo(character_ids)
pprint(embeddings['elmo_representations'][0].size())

'''
wget https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

pytorch_pretrained_bert convert_tf_checkpoint_to_pytorch \
  /home/dpappas/for_ryan/uncased_L-12_H-768_A-12/bert_model.ckpt \
  /home/dpappas/for_ryan/uncased_L-12_H-768_A-12/bert_config.json \
  /home/dpappas/for_ryan/uncased_L-12_H-768_A-12/pytorch_model.bin



sudo wget https://www.python.org/ftp/python/3.6.7/Python-3.6.7.tgz
tar -zxvf Python-3.6.7.tgz
cd Python-3.6.7
sudo ./configure --enable-optimizations 
sudo make altinstall

sudo pip3.6 install --upgrade pip
sudo pip3.6 install torch 
sudo pip3.6 install tqdm 
sudo pip3.6 install requests
sudo pip3.6 install allennlp
sudo pip3.6 install nltk
sudo pip3.6 install tensorflow
sudo pip3.6 install keras
sudo pip3.6 install boto3
sudo apt-get install lzma
sudo apt-get install liblzma-dev
sudo apt-get install python-lzma
sudo pip3.6 install gensim
sudo pip3.6 install python3-utils
sudo pip3.6 install pylzma
sudo pip3.6 install backports.lzma
sudo pip3.6 install patool
sudo apt-get install liblzma-doc
sudo pip3.6 install pyliblzma

'''

'''

def get_elmo_embeds(sentences):
    sentences       = [tokenize(s) for s in sentences if(len(s)>0)]
    character_ids   = batch_to_ids(sentences)
    embeddings      = elmo(character_ids)
    the_embeds      = embeddings['elmo_representations'][0]
    ret             = [
        the_embeds[i, :len(sentences[i]), :]
        for i in range(len(sentences))
    ]
    return ret

'''


