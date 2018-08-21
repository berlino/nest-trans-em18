import torch
import torch.nn.functional as F
from .stack_lstm import StackRNN
from .tree_rnn import LeafModule, ReduceModule, UnaryModule

class Stack(object):
    """
    Stack with vanilla recursive rnn
    Todo: add label to the process for debug
    """
    def __init__(self, cell, initial_state, reduce_linears, unary_linears, empty_vec):
        self.stack = StackRNN(cell, initial_state)

        # partial tree representations
        self.empty = (empty_vec, None)
        self.embeds = [self.empty]

        # vanilla recursive rnn
        self.reduce_linears = reduce_linears
        self.unary_linears = unary_linears

    def shift(self, word_vec):
        self.embeds.append((word_vec, None))
        self.stack.push(word_vec)
    
    def reduceX(self, label_id):
        self.stack.pop()
        self.stack.pop()
        embed_2, _ = self.embeds.pop()
        embed_1, _ = self.embeds.pop()

        new_embed = F.tanh(self.reduce_linears[label_id](torch.cat([embed_1, embed_2], 0)))
        self.embeds.append((new_embed, label_id))
        self.stack.push(new_embed)
   
    def unaryX(self, label_id):
        self.stack.pop()
        embed, _ = self.embeds.pop()
        new_embed = F.tanh(self.unary_linears[label_id](embed))
        self.embeds.append((new_embed, label_id))
        self.stack.push(new_embed)

    @DeprecationWarning
    def pop(self):
        """
        different from the pop operation of stack-lstm
        """
        assert len(self.stack) == 2
        element = self.stack.pop()
        return (element[0][0].squeeze(0), element[1])

    def embedding(self):
        stack_summary = self.stack.embedding()
        embed = self.embeds[-1][0]
        return (stack_summary, embed)

    def __len__(self):
        return len(self.stack) 

    def top3(self):
        t3 = self.stack.top3()
        assert len(t3) == 3
        return self.stack.top3()


class AugStack(object):
    """
    Stack with tree lstm
    """
    def __init__(self, cell, initial_state, ih_linear, oh_linear, reduce_linears, unary_linears, empty_vec):
        self.stack = StackRNN(cell, initial_state)

        # tree lstm
        self.leaf_module = LeafModule(ih_linear, oh_linear)
        self.reduce_module = ReduceModule(reduce_linears)
        self.unary_module = UnaryModule(unary_linears)

        # partial tree representations
        self.empty = ((empty_vec, empty_vec), None)
        self.embeds = [self.empty]


    def shift(self, word_vec):
        h, c = self.leaf_module.forward(word_vec)
        self.embeds.append(((h, c), None))
        self.stack.push(h)
    
    def reduceX(self, label_id):
        self.stack.pop()
        self.stack.pop()
        (h_2, c_2), _ = self.embeds.pop()
        (h_1, c_1), _ = self.embeds.pop()

        # (h_2, c_2) = self.stack.pop()
        # (h_1, c_1) = self.stack.pop()
        # self.embeds.pop()
        # self.embeds.pop()

        h, c = self.reduce_module.forward(c_1, h_1, c_2, h_2, label_id)
        self.embeds.append(((h, c), label_id))
        self.stack.push(h)
   
    def unaryX(self, label_id):
        self.stack.pop()
        (h_, c_), _ = self.embeds.pop()

        # (h_, c_ ) = self.stack.pop()
        # self.embeds.pop()

        h, c = self.unary_module.forward(c_, h_, label_id)
        self.embeds.append(((h, c), label_id))
        self.stack.push(h)

    def embedding(self):
        stack_summary = self.stack.embedding()
        h = self.embeds[-1][0][0]
        return (stack_summary, h)
    
    def top3(self):
        t3 = self.stack.top3()
        assert len(t3) == 3
        return self.stack.top3()

    def __len__(self):
        return len(self.stack) 
