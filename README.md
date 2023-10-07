# Cross-Domain-NER
```
usage: train.py [-h] -i INPUT_PATH -m METHOD --sent_vocab SENT_VOCAB --ner_tag_vocab NER_TAG_VOCAB --entity_tag_vocab ENTITY_TAG_VOCAB [--model_path MODEL_PATH] [--dropout_rate DROPOUT_RATE] [--embed_size EMBED_SIZE] [--hidden_size HIDDEN_SIZE]
                [--batch_size BATCH_SIZE] [--max_epoch MAX_EPOCH] [--clip_max_norm CLIP_MAX_NORM] [--lr LR] [--log_every LOG_EVERY] [--validation_every VALIDATION_EVERY] [--patience_threshold PATIENCE_THRESHOLD] [--max_patience MAX_PATIENCE] [--max_decay MAX_DECAY]
                [--lr_decay LR_DECAY] [--model_save_path MODEL_SAVE_PATH] [--optimizer_save_path OPTIMIZER_SAVE_PATH] [--cuda]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT_PATH, --input_path INPUT_PATH
                        path to input text
  -m METHOD, --method METHOD
                        Method to use - NER, NER_Entity, MOEE,Entity_MOEE, NER_ENTITY_MOEE
  --sent_vocab SENT_VOCAB
                        Path to sentence vocab json file
  --ner_tag_vocab NER_TAG_VOCAB
                        Path to NER tag vocab json file
  --entity_tag_vocab ENTITY_TAG_VOCAB
                        Path to Entity tag vocab json file
  --model_path MODEL_PATH
                        path to the saved model to load for evaluation
  --dropout_rate DROPOUT_RATE
                        dropout rate [default: 0.5]
  --embed_size EMBED_SIZE
                        size of word embedding [default: 256]
  --hidden_size HIDDEN_SIZE
                        size of hidden state [default: 256]
  --batch_size BATCH_SIZE
                        batch-size [default: 32]
  --max_epoch MAX_EPOCH
                        max epoch [default: 10]
  --clip_max_norm CLIP_MAX_NORM
                        clip max norm [default: 5.0]
  --lr LR               learning rate [default: 0.001]
  --log_every LOG_EVERY
                        log every [default: 10]
  --validation_every VALIDATION_EVERY
                        validation every [default: 250]
  --patience_threshold PATIENCE_THRESHOLD
                        patience threshold [default: 0.98]
  --max_patience MAX_PATIENCE
                        time of continuous worse performance to decay lr [default: 4]
  --max_decay MAX_DECAY
                        time of lr decay to early stop [default: 4]
  --lr_decay LR_DECAY   decay rate of lr [default: 0.5]
  --model_save_path MODEL_SAVE_PATH
                        model save path [default: ./model/model.pth]
  --optimizer_save_path OPTIMIZER_SAVE_PATH
                        optimizer save path [default: ./model/optimizer.pth]
  --cuda                use GPU

```
