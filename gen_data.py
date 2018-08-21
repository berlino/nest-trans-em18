#!/usr/bin/env python
from collections import defaultdict
from reader.reader import Reader
from config import config
import pickle
import re
import pdb


def batch_stat(batches):
    all_num = 0
    start_num = 0
    end_num = 0
    for token_batch, pos_batch, label_batch, action_batch in zip(*batches):
        for labels in label_batch:
            start_dic = defaultdict(list)
            end_dic = defaultdict(list)
            for ent in labels:
                start_dic[(ent[0], ent[2])].append(ent)
                end_dic[(ent[1], ent[2])].append(ent)
                all_num += 1
            for k,v in start_dic.items():
                if len(v) > 1:
                    start_num += len(v)
            for k,v in end_dic.items():
                if len(v) > 1:
                    end_num += len(v)
    
    print("All {}, start {}, end {}".format(all_num, start_num, end_num))


if __name__ == "__main__":
    reader = Reader(config)
    reader.read_all_data()

    # print reader.train_sents[0]
    train_batches, dev_batches, test_batches = reader.to_batch()
    f = open(config.data_path + "_train.pkl", 'wb')
    pickle.dump(train_batches, f)
    f.close()

    f = open(config.data_path + "_dev.pkl", 'wb')
    pickle.dump(dev_batches, f)
    f.close()

    f = open(config.data_path + "_test.pkl", 'wb')
    pickle.dump(test_batches, f)
    f.close()

    #batch_stat(train_batches)
    #batch_stat(dev_batches)
    #batch_stat(test_batches)

    # random_text
    random_batch = 1
    random_n = 1
    reader.debug_single_sample( 
        train_batches[0][random_batch][random_n],
        train_batches[1][random_batch][random_n], 
        train_batches[2][random_batch][random_n], 
        train_batches[3][random_batch][random_n], 
        )

    # misc config
    misc_dict = dict()
    misc_dict["voc_size"] = len(reader.word2id)
    misc_dict["pos_size"] = len(reader.pos2id)
    misc_dict["label_size"] = len(reader.label2id)

    misc_dict["id2word"] = reader.id2word
    misc_dict["id2pos"] = reader.id2pos
    misc_dict["id2action"] = reader.id2action
    misc_dict["id2label"] = reader.id2label
    f = open(config.data_path + "_config.pkl", 'wb')
    pickle.dump(misc_dict, f)
    f.close()

    if config.pre_trained:
        reader.gen_vectors_glove()
