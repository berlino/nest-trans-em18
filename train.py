#!/usr/bin/env python
import torch
import copy
import time
import numpy as np
import pickle
from config import config
import random
from random import shuffle
from util.evaluate import evaluate, get_f1, get_action_acc
from model.transition_forest import TransitionForest
from training.util import adjust_learning_rate, clip_model_grad, create_opt, load_dynamic_config
import pdb

# load data
f = open(config.data_path + "_train.pkl", 'rb')
train_token_batches, train_pos_batches, train_label_batches, train_action_batches = pickle.load(f)
f.close()
f = open(config.data_path + "_dev.pkl", 'rb')
dev_token_batches, dev_pos_batches, dev_label_batches, dev_action_batches = pickle.load(f)
f.close()
f = open(config.data_path + "_test.pkl", 'rb')
test_token_batches, test_pos_batches, test_label_batches, test_action_batches = pickle.load(f)
f.close()

# misc info  
# TODO: get it better
misc_config = pickle.load(open(config.data_path + "_config.pkl", 'rb'))
load_dynamic_config(misc_config, config)
config.id2label = misc_config["id2label"]
config.id2action = misc_config["id2action"]
config.id2word = misc_config["id2word"]
config.id2pos = misc_config["id2pos"]


# ner_model = TransitionMR(config)
ner_model = TransitionForest(config)
# ner_model.rand_init()
if config.pre_trained:
    ner_model.load_vector()
if config.if_gpu and torch.cuda.is_available(): ner_model = ner_model.cuda()

parameters = filter(lambda p: p.requires_grad, ner_model.parameters())
optimizer = create_opt(parameters, config)

print("{0} batches expected for training".format(len(train_token_batches)))
best_model = None
best_per = 0
train_all_batches = list(zip(train_token_batches, train_pos_batches, train_label_batches, train_action_batches))
if config.if_shuffle:
    shuffle(train_all_batches)


train_start_time = time.time()
early_counter = 0
decay_counter = 0
for e_ in range(config.epoch):
    print("Epoch: ", e_ + 1)
    batch_counter = 0
    for token_batch, pos_batch, label_batch, action_batch in train_all_batches:
        batch_len = len(token_batch)
        sent_len = len(token_batch[0])

        token_batch_var = torch.LongTensor(np.array(token_batch))
        pos_batch_var = torch.LongTensor(np.array(pos_batch))
        action_batch_var = torch.LongTensor(np.array(action_batch))
        if config.if_gpu:
            token_batch_var = token_batch_var.cuda()
            pos_batch_var = pos_batch_var.cuda()
            action_batch_var = action_batch_var.cuda()

        ner_model.train()
        optimizer.zero_grad()
        loss, (preds, correct_nums, total_nums) = ner_model.forward(token_batch_var, pos_batch_var, label_batch, action_batch_var)
        if type(loss) != float: 
            loss.backward()
            clip_model_grad(ner_model, config.clip_norm)
            action_acc = get_action_acc(correct_nums, total_nums)
            print("batch {0} with {1} instance and sentece length {2} loss {3}, action acc {4}".format(
                batch_counter, batch_len, sent_len, loss.cpu().data.numpy().item(), action_acc))

            optimizer.step()

        batch_counter += 1

    if (e_+1) % config.check_every != 0:
        continue

    # evaluating dev and always save the best
    cur_time = time.time()
    dev_batch_zip = zip(dev_token_batches, dev_pos_batches, dev_label_batches, dev_action_batches)
    # dev_batch_zip = zip(train_token_batches, train_pos_batches, train_label_batches, train_action_batches)
    f1 = get_f1(ner_model, dev_batch_zip, config)
    print("Dev step took {} seconds".format(time.time() - cur_time))

    # early stop
    if f1 > best_per:
        early_counter = 0
        best_per = f1
        del best_model
        best_model = copy.deepcopy(ner_model)
    else:
        early_counter += 1
        if early_counter > config.lr_patience:
            decay_counter += 1
            early_counter = 0
            if decay_counter > config.decay_patience:
                break
            else:
                adjust_learning_rate(optimizer)
print("")
print("Training step took {} seconds".format(time.time() - train_start_time))
print("Best dev acc {0}".format(best_per))
print("")

# remember to eval after loading the model. for the reason of batchnorm and dropout
cur_time = time.time()
test_batch_zip = zip(test_token_batches, test_pos_batches, test_label_batches, test_action_batches)
f1 = get_f1(best_model, test_batch_zip, config)
print("Test step took {} seconds".format(time.time() - cur_time))

serial_number = str(random.randint(0,248))
this_model_path = config.model_path + "_" + serial_number
print("Dumping model to {0}".format(this_model_path))
torch.save(best_model.state_dict(), this_model_path)
