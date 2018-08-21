from __future__ import division


class Config():
    def __init__(self):
        self.root_path = "./"

        # for data loader
        self.data_set = "genia_sample"
        self.batch_size = 64
        self.if_shuffle = True

        # override when loading data
        self.voc_size = None
        self.pos_size = None
        self.label_size = None
        self.actions = None

        # embed size
        self.token_embed = 100
        self.action_embed = 20
        self.entity_embed = self.action_embed
        self.pos_embed = 20
        self.input_dropout = 0.5
        self.lstm_dropout = 0.5

        # for lstm
        self.if_treelstm = True
        self.rnn_layers = 1
        self.hidden_dim = 128

        # reversed, for convenience of buffer
        self.reversed = True

        # for training
        self.embed_path = self.root_path + "/data/word_vec_{0}_{1}.pkl".format(self.data_set, self.token_embed)
        self.epoch = 500
        self.if_gpu = False
        self.opt = "Adam"
        self.lr = 0.005 # [0.3, 0.00006]
        self.l2 = 1e-4
        self.check_every = 1
        self.clip_norm = 3

        # for early stop
        self.lr_patience = 6
        self.decay_patience = 3

        self.pre_trained = True
        self.data_path = self.root_path + "/data/{0}".format(self.data_set)
        self.model_path = self.root_path + "/dumps/{0}_model.pt".format(self.data_set)



    def __repr__(self):
        return str(vars(self))


config = Config()
