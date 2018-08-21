import logging
import pickle
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import pdb

import model.util as util
from module.stack_lstm import StackRNN, Buffer, ShiftRNN, BufferR
from util.oracle import Executor

class TransitionMR(nn.Module):
    def __init__(self, config):
        super(TransitionMR, self).__init__()
        self.config = config

        # for debug
        self.id2word = config.id2word
        self.id2label = config.id2label
        self.id2action = config.id2action
        self.id2pos = config.id2pos
        self.action2id = {k:i for i,k in enumerate(self.id2action)}
        self.label2id = {k:i for i,k in enumerate(self.id2label)}
        
        self.word_embeds = nn.Embedding(config.voc_size, config.token_embed)
        self.pos_embeds = nn.Embedding(config.pos_size, config.pos_embed)
        self.action_embeds = nn.Embedding(len(config.id2action), config.action_embed)
        self.entity_embeds = nn.Embedding(len(config.id2label) + 1, config.entity_embed)

        self.rnn_layers = config.rnn_layers
        self.token_lstm = nn.LSTM(config.token_embed + config.pos_embed, config.hidden_dim, 
            num_layers=config.rnn_layers, bidirectional=False, batch_first=True)

        self.stack_lstm = nn.LSTMCell(config.hidden_dim, config.hidden_dim)
        self.stack_lstm_initial = nn.ParameterList([util.xavier_init(self.config.if_gpu, 1, self.config.hidden_dim), 
            util.xavier_init(self.config.if_gpu, 1, self.config.hidden_dim)])
        self.stack_empty = nn.Parameter(torch.randn(config.hidden_dim))

        self.output_lstm = nn.LSTMCell(config.hidden_dim + config.entity_embed , config.hidden_dim)
        self.output_lstm_initial = nn.ParameterList([util.xavier_init(self.config.if_gpu, 1, self.config.hidden_dim), 
            util.xavier_init(self.config.if_gpu, 1, self.config.hidden_dim)])
        self.output_empty = nn.Parameter(torch.randn(config.hidden_dim))

        self.action_lstm = nn.LSTMCell(config.action_embed, config.hidden_dim)
        self.action_lstm_initial = nn.ParameterList([util.xavier_init(self.config.if_gpu, 1, self.config.hidden_dim), 
            util.xavier_init(self.config.if_gpu, 1, self.config.hidden_dim)])
        self.action_empty = nn.Parameter(torch.randn(config.hidden_dim))

        self.token_empty = nn.Parameter(torch.randn(config.hidden_dim))
        self.token_lstm_empty = nn.Parameter(torch.randn(config.hidden_dim))

        self.shift2stack = nn.Linear(config.hidden_dim + config.token_embed + config.pos_embed, config.hidden_dim)
        self.reduce2stack = nn.ModuleList([nn.Linear( 3 * config.hidden_dim,
            config.hidden_dim) for _ in range(len(self.id2label) * 2)] )
        self.unary2stack = nn.ModuleList([nn.Linear( 2 * config.hidden_dim,
            config.hidden_dim) for _ in range(len(self.id2label))] )

        self.input_dropout = nn.Dropout(p=config.input_dropout)
        self.lstm_dropout = nn.Dropout(p=config.lstm_dropout)

        self.hidden2feat = nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        self.feat2act = nn.Linear(config.hidden_dim, len(config.id2action))
        self.entity2output = nn.Linear(config.hidden_dim * 2 + config.action_embed, config.token_embed)

        self.executor = Executor(label2id=self.label2id)
    

    def action2label_star(self, action_id):
        """
        include label*
        label* = label + label_size
        """
        label_str = self.id2action[action_id].split("-")[1]
        if label_str[-1] == "*":
            return self.label2id[label_str[:-1]] + len(self.id2label)
        else:
            return self.label2id[label_str]

    def get_possible_actions(self, stack, buffer):
        valid_actions = []
        cur_state_label_id = stack.embedding()[1]

        if len(buffer) > 0:
            valid_actions.append(self.action2id["Shift"])
        if len(stack) > 0 and cur_state_label_id is None:
            valid_actions += [self.action2id["Unary-"+label] for label in self.id2label]
        if len(stack) == 2 and len(buffer) == 0:
            valid_actions += [self.action2id["Reduce-" + label] for label in self.id2label]
        elif len(stack) > 1:
            valid_actions += [self.action2id["Reduce-" + label] for label in self.id2label]
            valid_actions += [self.action2id["Reduce-" + label + "*"] for label in self.id2label]
        if len(stack) == 1 and (cur_state_label_id is None or cur_state_label_id < len(self.id2label)):  # cannot pop intermedia label
            valid_actions.append(self.action2id["Pop"])

        valid_actions_v = Variable(torch.LongTensor(valid_actions))
        if self.config.if_gpu:  valid_actions_v = valid_actions_v.cuda()
        return valid_actions, valid_actions_v

    def load_vector(self):
        with open(self.config.embed_path, 'rb') as f:
            vectors = pickle.load(f)
            t_v = torch.Tensor(vectors)
            print("Loading from {} with size {}".format(self.config.embed_path, t_v.size()))
            self.word_embeds.weight = nn.Parameter(t_v)
            # self.word_embeds.weight.requires_grad = False

    def rand_init(self):
        """
        TODO: to be updated
        """
        util.init_embedding(self.word_embeds.weight)
        util.init_embedding(self.pos_embeds.weight)
        util.init_embedding(self.action_embeds.weight)
        util.init_embedding(self.entity_embeds.weight)
            
        util.init_linear(self.hidden2feat)
        util.init_linear(self.feat2act)
        util.init_linear(self.entity2output)
        
        util.init_lstm_cell(self.stack_lstm)
        util.init_lstm_cell(self.output_lstm)

    
    def _map2actions(self, action_v):
        """
        for debugging
        """
        if type(action_v) == torch.Tensor:
            action_v = action_v.cpu().data.numpy()
        return [self.id2action[a] for a in action_v]


    def forward(self, token_batch, pos_batch, label_batch, action_batch):
        batch_size, sent_len = token_batch.size()
        _, action_max_len = action_batch.size()
        word_mat = self.word_embeds(token_batch)
        pos_mat = self.pos_embeds(pos_batch)
        token_mat = self.input_dropout(torch.cat([word_mat, pos_mat], 2))

        losses = [[] for i in range(batch_size)]
        predict_actions = [[] for i in range(batch_size)]

        correct_num_action = [0 for i in range(batch_size)]
        totol_num_action = [0 for i in range(batch_size)]
        
        tok_output, _ = self.token_lstm(token_mat) 

        for batch_idx in range(batch_size):
            stack = ShiftRNN(self.stack_lstm, self.stack_lstm_initial, self.shift2stack, self.reduce2stack, self.unary2stack, self.stack_empty)
            output_stack = StackRNN(self.output_lstm, self.output_lstm_initial, self.output_empty)
            action_stack = StackRNN(self.action_lstm, self.action_lstm_initial, self.action_empty)

            token_buffer = BufferR(token_mat[batch_idx], self.token_empty)
            token_lstm_buffer = BufferR(tok_output[batch_idx], self.token_lstm_empty)

            for action_idx in range(action_max_len):
                cur_action = action_batch.data[batch_idx][action_idx]
                if type(cur_action) != int: cur_action = cur_action.cpu().data.numpy().item()
                if cur_action == self.action2id["Pad"]:  break
                real_action = self.id2action[cur_action]

                #compute the loss
                if type(cur_action) != int: cur_action = cur_action.cpu().data.numpy().item()
                valid_actions, valid_actions_v = self.get_possible_actions(stack, token_buffer)
                log_probs = None
                if len(valid_actions) > 1:
                    decision_feature = torch.cat([stack.embedding()[0], output_stack.embedding(), 
                        token_lstm_buffer.embedding(), action_stack.embedding()], 0)
                    hidden_output = torch.tanh(self.hidden2feat(self.input_dropout(decision_feature)))

                    logits = self.feat2act(hidden_output)[valid_actions_v]
                    va_table = {a: i for i, a in enumerate(valid_actions)}
                    log_probs = torch.nn.functional.log_softmax(logits, 0)
                    max_id = torch.max(log_probs.cpu(), 0)[1].data.numpy().item()
                    action_predict = valid_actions[max_id]

                    if cur_action not in va_table:
                        pdb.set_trace()

                    if log_probs is not None:
                        single_loss = log_probs[va_table[cur_action]]
                        # losses[batch_idx].append(single_loss)

                        # for balance
                        if real_action == "Pop":
                            losses[batch_idx].append(self.config.discount * single_loss)
                        else:
                            losses[batch_idx].append(single_loss)
                else:
                    action_predict = valid_actions[0]

                predict_actions[batch_idx].append(action_predict)
                totol_num_action[batch_idx] += 1
                if real_action == self.id2action[action_predict]:
                    correct_num_action[batch_idx] += 1
                
                # push the action
                action_v = Variable(torch.LongTensor([cur_action]))
                if self.config.if_gpu:  action_v = action_v.cuda()
                action_var = self.action_embeds(action_v).squeeze(0)
                action_var = self.input_dropout(action_var)
                action_stack.push(action_var)

                # execute the action
                if real_action == 'Shift':
                    buffer_state = token_lstm_buffer.pop()
                    word_state = token_buffer.pop() 
                    stack.shift(buffer_state, word_state)
                elif real_action.startswith('Reduce'):
                    buffer_state = token_lstm_buffer.embedding()
                    label_id = self.action2label_star(cur_action)
                    stack.reduceX(buffer_state, label_id)
                elif real_action.startswith('Unary'):
                    buffer_state = token_lstm_buffer.embedding()
                    label_id = self.action2label_star(cur_action)
                    stack.unaryX(buffer_state, label_id)
                elif real_action == 'Pop':
                    label_id = stack.embedding()[1]
                    if label_id is None:  label_id = len(self.id2label)
                    assert label_id <= len(self.id2label)
                    entity_id_v = Variable(torch.LongTensor([label_id]))
                    if self.config.if_gpu:  entity_id_v = entity_id_v.cuda()
                    entity_vec = self.entity_embeds(entity_id_v).squeeze(0)
                    entity_vec = self.input_dropout(entity_vec)

                    stack_state, _ = stack.pop()
                    entity_state = torch.cat([stack_state, entity_vec], 0)
                    output_stack.push(entity_state)
                else:
                    raise ValueError
            
            # sanity check
            assert len(token_buffer) == 0
            assert len(token_lstm_buffer) == 0
            action_strs = [self.id2action[a] for a in predict_actions[batch_idx]]
            # print(" ".join(action_strs))

        loss_v = []
        action_num = 0
        for idx in range(batch_size):
            loss_v.append(sum(losses[idx]))
            action_num += len(losses[idx])
    
        batch_mean_loss = -1.0 * sum(loss_v) / batch_size
        return batch_mean_loss, (predict_actions, correct_num_action, totol_num_action) if len(losses) > 0 else None
    
    def predict(self, token_batch, pos_batch):
        batch_size, sent_len = token_batch.size()
        word_mat = self.word_embeds(token_batch)
        pos_mat = self.pos_embeds(pos_batch)
        token_mat = torch.cat([word_mat, pos_mat], 2)

        losses = [[] for i in range(batch_size)]
        correct_num_action = [0 for i in range(batch_size)]
        predict_actions = [[] for i in range(batch_size)]
        
        tok_output, _ = self.token_lstm(token_mat) 

        for batch_idx in range(batch_size):
            stack = ShiftRNN(self.stack_lstm, self.stack_lstm_initial, self.shift2stack, self.reduce2stack, self.unary2stack, self.stack_empty)
            output_stack = StackRNN(self.output_lstm, self.output_lstm_initial, self.output_empty)
            action_stack = StackRNN(self.action_lstm, self.action_lstm_initial, self.action_empty)

            token_buffer = BufferR(token_mat[batch_idx], self.token_empty)
            token_lstm_buffer = BufferR(tok_output[batch_idx], self.token_lstm_empty)

            while len(token_buffer) > 0 or len(stack) > 0:
                # pdb.set_trace()
                valid_actions, valid_actions_v = self.get_possible_actions(stack, token_buffer)
                if len(valid_actions) > 1:
                    decision_feature = torch.cat([stack.embedding()[0], output_stack.embedding(), 
                        token_lstm_buffer.embedding(), action_stack.embedding()], 0)
                    hidden_output = torch.tanh(self.hidden2feat(decision_feature))

                    logits = self.feat2act(hidden_output)[valid_actions_v]
                    va_table = {a: i for i, a in enumerate(valid_actions)}
                    log_probs = torch.nn.functional.log_softmax(logits, 0)
                    max_id = torch.max(log_probs.cpu(), 0)[1].data.numpy().item()
                    action_predict = valid_actions[max_id]
                else:
                    if len(valid_actions) == 0: pdb.set_trace()
                    action_predict = valid_actions[0]

                predict_actions[batch_idx].append(self.id2action[action_predict])
                real_action = self.id2action[action_predict]

                # push the action
                action_v = Variable(torch.LongTensor([action_predict]))
                if self.config.if_gpu:  action_v = action_v.cuda()
                action_var = self.action_embeds(action_v).squeeze(0)
                action_stack.push(action_var)
                
                # execute the action
                if real_action == 'Shift':
                    buffer_state = token_lstm_buffer.pop()
                    word_state = token_buffer.pop() 
                    stack.shift(buffer_state, word_state)
                elif real_action.startswith('Reduce'):
                    buffer_state = token_lstm_buffer.embedding()
                    label_id = self.action2label_star(action_predict)
                    stack.reduceX(buffer_state, label_id)
                elif real_action.startswith('Unary'):
                    buffer_state = token_lstm_buffer.embedding()
                    label_id = self.action2label_star(action_predict)
                    stack.unaryX(buffer_state, label_id)
                elif real_action == 'Pop':
                    label_id = stack.embedding()[1]
                    if label_id is None:  label_id = len(self.id2label)
                    assert label_id <= len(self.id2label)
                    entity_id_v = Variable(torch.LongTensor([label_id]))
                    if self.config.if_gpu:  entity_id_v = entity_id_v.cuda()
                    entity_vec = self.entity_embeds(entity_id_v).squeeze(0)
                    
                    stack_state, _ = stack.pop()
                    entity_state = torch.cat([stack_state, entity_vec], 0)
                    output_stack.push(entity_state)
                else:
                    raise ValueError

            # sanity check
            assert len(token_buffer) == 0
            assert len(token_lstm_buffer) == 0
            # print(" ".join(predict_actions[batch_idx]))
                
        ret_triples = []
        for i, actions in enumerate(predict_actions):
            triples = self.executor.execute(sent_len, actions)
            ret_triples.append(triples)

        return ret_triples

