import torch
import torch.nn as nn
import torch.nn.functional as F

@DeprecationWarning
class TreeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, reduce_linears, unary_linears):
        """
        linears: 4 * 2label_size 
        """
        super(TreeLSTM, self).__init__()

        self.leaf_module = LeafModule(input_dim, hidden_dim)
        self.reduce_module = ReduceModule(reduce_linears)
        self.unary_moduel = UnaryModule(unary_linears)
    
    def forward(self, node_str, vecs):
        if node_str == "leaf":
            return self.leaf_module.forward(vecs)
        elif node_str == "reduce":
            return self.reduce_module.forward(*vecs)
        elif node_str == "unary":
            return self.unary_moduel.forward(*vecs)
        else:
            raise ValueError

class LeafModule(nn.Module):
    """
    return hidden_vec, cell_vec
    """
    def __init__(self, w_ixh, w_oxh):
        super(LeafModule, self).__init__()
        self.w_ixh = w_ixh
        self.w_oxh = w_oxh
    
    def forward(self, input_vec):
        c = self.w_ixh(input_vec)
        o = F.sigmoid(self.w_oxh(input_vec))
        h = o * F.tanh(c)
        return h, c

class ReduceModule(nn.Module):
    def __init__(self, reduce_linears):
        super(ReduceModule, self).__init__()
        self.reduce_linears = reduce_linears
    
    def forward(self, lc, lh , rc, rh, label):
        lrh = torch.cat([lh, rh], 0)
        i = F.sigmoid(self.reduce_linears[0][label](lrh))
        lf = F.sigmoid(self.reduce_linears[1][label](lrh))
        rf = F.sigmoid(self.reduce_linears[2][label](lrh))
        update = F.tanh(self.reduce_linears[3][label](lrh))
        c =  i * update + lf*lc + rf*rc
        h = F.tanh(c)
        return h, c

class UnaryModule(nn.Module):
    def __init__(self, unary_linears):
        super(UnaryModule, self).__init__()
        self.unary_linears = unary_linears
    
    def forward(self, c, h, label):
        i = F.sigmoid(self.unary_linears[0][label](h))
        f = F.sigmoid(self.unary_linears[1][label](h))
        update = F.tanh(self.unary_linears[2][label](h))
        c =  i * update + f * c
        h = F.tanh(c)
        return h, c