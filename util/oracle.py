from collections import defaultdict
import pdb


class SegmentSet(object):
    """
    maintain the outermost constituents
    """
    def __init__(self, tuples):
        """
        tuple: <start, end, ...>
        """
        self.tuples = tuples
        self.query_cache = {}
    
    def __getitem__(self, q):
        """
        return tuples that overlaps with the input
        """
        rets = []
        for item in self.tuples:
            if ( q[0] >= item[0] and q[0] <= item[1]) or (( q[1] >= item[0] and q[1] <= item[1]))\
                or ( q[0] < item[0] and q[1] > item[1]):
                if q != item:
                    rets.append(item)
        return rets
    
    def query(self, idx):
        """
        if idx is contained in a chunk
        """
        if len(self.query_cache) == 0:
            for item in self.tuples:
                for i in range(item[0], item[1] + 1):
                    self.query_cache[i] = item
        if idx in self.query_cache:
            return self.query_cache[idx]
        else:
            return None
    
    def update(self, parent, children):
        """
        update the segment set, remove children from tuples
        """
        for item in children:
            self.tuples.remove(item)
        self.tuples.append(parent)

class TreeNode(object):
    def __init__(self, segment, label, children):
        self.segment = segment
        self.label = label
        self.children = children
    
    def __repr__(self):
        return "<start: {}, end: {}, label: {}>".format(self.segment[0], self.segment[1], self.label)

class Executor(object):
    """
    Shift, Reduce, Pop
    """
    def __init__(self, labels=None, label2id=None):
        ACTIONS_1 = ["Shift", "Pop"]
        ACTIONS_2 = ["Unary-", "Reduce-"]
        # self.actions = ACTIONS_1 + [ac_ + label_ for ac_ in ACTIONS_2 for label_ in labels]
        self.non_label = "O"
        self.label2id = label2id

    def execute(self, sent_len, actions):
        """
        maps a sequence of actions to triples
        """
        output = []
        stack = []
        buffer_len = sent_len
        buffer_pointer = 0

        for action in actions:
            if action == "Shift":
                item = TreeNode((buffer_pointer, buffer_pointer), self.non_label, None)
                stack.append(item)
                buffer_pointer += 1
            elif action == "Pop":
                if len(stack) != 1: pdb.set_trace()
                item = stack[:][0]
                output.append(item)
                stack = []
            elif action.startswith("Reduce"):
                label = action.split("-")[1]
                if len(stack) < 2: pdb.set_trace()
                x1 = stack.pop()
                x0 = stack.pop()
                new_segment = (x0.segment[0], x1.segment[1])
                item = TreeNode(new_segment, label, (x0, x1))
                stack.append(item)
            elif action.startswith("Unary"):
                label = action.split("-")[1]
                stack[-1].label = label
        
        ret_triples = []
        def recur_resolve(treenode):
            if treenode.label != self.non_label and treenode.label[-1] != "*":
                if self.label2id is not None:
                    ret_triples.append((treenode.segment[0], treenode.segment[1], self.label2id[treenode.label]))
                else:
                    ret_triples.append((treenode.segment[0], treenode.segment[1], treenode.label))

            if treenode.children is not None:
                for child in treenode.children:
                    recur_resolve(child) 
        for node in output:
            recur_resolve(node)
        return ret_triples
    

    def triple2actions(self, triples, sent_len, branching="left"):
        """
        Convert triples to a sequence of actions 
        """
        sort_triples = sorted(triples, key=lambda x: x[1] - x[0])

        chunk2actions = defaultdict(list)
        segmentset = SegmentSet([])
        for triple in sort_triples:
            # print(triple)
            if triple[1] - triple[0] == 0:
                chunk2actions[triple[0], triple[1]].append("Shift")
                chunk2actions[triple[0], triple[1]].append("Unary-" + triple[2])
                segmentset.update(triple, [])
            else:
                # pdb.set_trace()
                overlaps = segmentset[triple]
                segmentset.update(triple, overlaps)
                if len(overlaps) == 0:
                    chunk2actions[triple[0], triple[1]].append("Shift")
                    for _ in range(triple[0], triple[1]-1):
                        chunk2actions[triple[0], triple[1]].append("Shift")
                        chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2] + "*")
                    chunk2actions[triple[0], triple[1]].append("Shift")
                    chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2])
                else:
                    # pdb.set_trace()
                    # serve as a stack
                    overlaps = sorted(overlaps, key=lambda x: -x[0])
                    i = triple[0]
                    stack_empty = True
                    while i < triple[1] + 1:
                        if len(overlaps) > 0:
                            # before overlap
                            count = overlaps[-1][0] - i
                            if count > 0:
                                if stack_empty:
                                    chunk2actions[triple[0], triple[1]].append("Shift")
                                    num_shift_reduce = count - 1
                                else:
                                    num_shift_reduce = count
                            
                                if num_shift_reduce > 0:
                                    for _ in range(num_shift_reduce):
                                        chunk2actions[triple[0], triple[1]].append("Shift")
                                        chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2] + "*")

                            # overlap
                            chunk2actions[triple[0], triple[1]] += chunk2actions[overlaps[-1][0], overlaps[-1][1]]

                            # after overlap
                            if triple[1] == overlaps[-1][1]:
                                chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2])
                            elif (overlaps[-1][0] - triple[0]) > 0:
                                chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2] + "*")

                            # pop it
                            i = overlaps[-1][1] + 1
                            del chunk2actions[overlaps[-1][0], overlaps[-1][1]]
                            stack_empty = False
                            overlaps.pop()
                        else:
                            count = triple[1] - i + 1
                            if count > 1:
                                for _ in range(count-1):
                                    chunk2actions[triple[0], triple[1]].append("Shift")
                                    chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2] + "*")
                            chunk2actions[triple[0], triple[1]].append("Shift")
                            chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2])
                            i += count

        ret_actions = []
        i = 0
        while i < sent_len:
            seg = segmentset.query(i)
            if seg is None:
                ret_actions.append("Shift")
                ret_actions.append("Pop")
                i += 1
            else:
                ret_actions += chunk2actions[seg[0], seg[1]]
                ret_actions.append("Pop")
                i = seg[1] + 1
        return ret_actions
                
class ExecutorR(object):
    """
    Shift, Reduce
    """
    def __init__(self, labels=None, label2id=None, mode="normal"):
        ACTIONS_1 = ["Shift"]
        ACTIONS_2 = ["Unary-", "Reduce-"]
        self.non_label = "O"
        self.label2id = label2id
        self.mode = mode

    def execute(self, sent_len, actions):
        """
        maps a sequence of actions to triples
        return: valid_triples, num_invalid_triples, 
        """
        stack = []
        buffer_len = sent_len
        buffer_pointer = 0

        for action in actions:
            if action == "Shift":
                item = TreeNode((buffer_pointer, buffer_pointer), self.non_label, None)
                stack.append(item)
                buffer_pointer += 1
            elif action.startswith("Reduce"):
                label = action.split("-")[1]
                if len(stack) < 2: pdb.set_trace()
                x1 = stack.pop()
                x0 = stack.pop()
                new_segment = (x0.segment[0], x1.segment[1])
                item = TreeNode(new_segment, label, (x0, x1))
                stack.append(item)
            elif action.startswith("Unary"):
                label = action.split("-")[1]
                stack[-1].label = label
        
        num_invalid = 0
        ret_triples = []
        def recur_resolve(treenode):
            if treenode.label != self.non_label and treenode.label[-1] != "*":
                if self.label2id is not None:
                    ret_triples.append((treenode.segment[0], treenode.segment[1], self.label2id[treenode.label]))
                else:
                    ret_triples.append((treenode.segment[0], treenode.segment[1], treenode.label))
            elif treenode.label == self.non_label and self.mode == "mws":
                if self.label2id is not None:
                    ret_triples.append((treenode.segment[0], treenode.segment[1], self.label2id["0"]))
                else:
                    ret_triples.append((treenode.segment[0], treenode.segment[1], "0"))


            if treenode.children is not None:
                for child in treenode.children:
                    recur_resolve(child) 

        for node in stack:
            # invalid nodes
            if node.label != self.non_label and node.label[-1] == "*":
                num_invalid += 0
    
            recur_resolve(node)
        return ret_triples, num_invalid
    

    def triple2actions(self, triples, sent_len, branching="left"):
        """
        Convert triples to a sequence of actions 
        """

        # filter length with 1
        if self.mode == "mws":
            filter_triples = []
            for t in triples:
                if t[0] != t[1]:
                    filter_triples.append(t)
            triples = filter_triples

        sort_triples = sorted(triples, key=lambda x: x[1] - x[0])

        chunk2actions = defaultdict(list)
        segmentset = SegmentSet([])
        for triple in sort_triples:
            # print(triple)
            if triple[1] - triple[0] == 0:
                chunk2actions[triple[0], triple[1]].append("Shift")
                chunk2actions[triple[0], triple[1]].append("Unary-" + triple[2])
                segmentset.update(triple, [])
            else:
                # pdb.set_trace()
                overlaps = segmentset[triple]
                segmentset.update(triple, overlaps)
                if len(overlaps) == 0:
                    chunk2actions[triple[0], triple[1]].append("Shift")
                    for _ in range(triple[0], triple[1]-1):
                        chunk2actions[triple[0], triple[1]].append("Shift")
                        chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2] + "*")
                    chunk2actions[triple[0], triple[1]].append("Shift")
                    chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2])
                else:
                    # pdb.set_trace()
                    # serve as a stack
                    overlaps = sorted(overlaps, key=lambda x: -x[0])
                    i = triple[0]
                    stack_empty = True
                    while i < triple[1] + 1:
                        if len(overlaps) > 0:
                            # before overlap
                            count = overlaps[-1][0] - i
                            if count > 0:
                                if stack_empty:
                                    chunk2actions[triple[0], triple[1]].append("Shift")
                                    num_shift_reduce = count - 1
                                else:
                                    num_shift_reduce = count
                            
                                if num_shift_reduce > 0:
                                    for _ in range(num_shift_reduce):
                                        chunk2actions[triple[0], triple[1]].append("Shift")
                                        chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2] + "*")

                            # overlap
                            chunk2actions[triple[0], triple[1]] += chunk2actions[overlaps[-1][0], overlaps[-1][1]]

                            # after overlap
                            if triple[1] == overlaps[-1][1]:
                                chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2])
                            elif (overlaps[-1][0] - triple[0]) > 0:
                                chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2] + "*")

                            # pop it
                            i = overlaps[-1][1] + 1
                            del chunk2actions[overlaps[-1][0], overlaps[-1][1]]
                            stack_empty = False
                            overlaps.pop()
                        else:
                            count = triple[1] - i + 1
                            if count > 1:
                                for _ in range(count-1):
                                    chunk2actions[triple[0], triple[1]].append("Shift")
                                    chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2] + "*")
                            chunk2actions[triple[0], triple[1]].append("Shift")
                            chunk2actions[triple[0], triple[1]].append("Reduce-" + triple[2])
                            i += count

        ret_actions = []
        i = 0
        while i < sent_len:
            seg = segmentset.query(i)
            if seg is None:
                ret_actions.append("Shift")
                i += 1
            else:
                ret_actions += chunk2actions[seg[0], seg[1]]
                i = seg[1] + 1
        return ret_actions

if __name__ == "__main__":
    executor = Executor(["Person", "Location", "GPE"])
    # executor2 = ExecutorR(["Person", "Location", "GPE"])
    executor2 = ExecutorR(["0"], mode="mws")
    sentence = "I liked San Antonio City very much .".split()
    # triples = [(2, 4, "Person"), (2, 2, "Person"), (3, 3, "Person")]
    # triples = []
    triples = [(0, 4, "0"), (2, 3, "0")]
    actions = executor2.triple2actions(triples, len(sentence))
    print(actions)
    triples = executor2.execute(len(sentence), actions)
    print(triples)
