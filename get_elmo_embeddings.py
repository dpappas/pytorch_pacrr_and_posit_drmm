


# download the files
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ccTfSBNofHIBCsZpHR2YU_7l86UMR460' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ccTfSBNofHIBCsZpHR2YU_7l86UMR460" -O weights.hdf5 && rm -rf /tmp/cookies.txt
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1W8ZXsMNxdq4s-KdgoL42ivSG49VOKRze' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1W8ZXsMNxdq4s-KdgoL42ivSG49VOKRze" -O options.json && rm -rf /tmp/cookies.txt


from allennlp.modules.elmo import Elmo, batch_to_ids
options_file    = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
weight_file     = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
elmo            = Elmo(options_file, weight_file, 2, dropout=0)

# use batch_to_ids to convert sentences to character ids
sentences       = [['First', 'sentence', '.'], ['Another', '.']]
character_ids   = batch_to_ids(sentences)
embeddings      = elmo(character_ids)





