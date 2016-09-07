from optparse import OptionParser
from collections import defaultdict
import sys
import bisect

def binary_search_left(arr, target):
    idx = bisect.bisect_left(arr, target)
    if idx == len(arr):
        return False, idx
    return arr[idx] == target, idx

def read_data(path):
    mat = []
    with open(path, 'r') as f_r:
        for line in f_r:
            line = line.strip().decode("utf-8")
            parts = line.split(',')
            mat.append(parts)
    return mat

class Tree_Node:
    '''
        1.This is for tree node, has parent node, children nodes array, and next node to
        point to the same item in the tree
        2.val is the transaction item
        3.children array is sorted by alphabet order
        4.count by scan the transaction recorder
        5.for the next node, there are lots same value node in the tree, how do we select the next node?

    '''
    def __init__(self, key):
        self.parent = None
        self.children = []
        self.count = 1
        self.key = key
        self.next = None

    def __str__(self):
        return str(self.key)

    def __eq__(self, other):
        return self.key == other.key

    def __cmp__(self, other):
        return cmp(self.key, other.key)

    def __lt__(self, other):
        return self.key < other.key

    def __gt__(self, other):
        return self.key > other.key

class Head_Table_Item:
    '''
        1.using linked list to form the head table
        2.count item
        3.next point to the same node in the tree
    '''
    def __init__(self, key, node):
        self.key = key
        self.next = node
        self.count = 0
        if node is not None:
            self.count = node.count

    def __eq__(self, other):
        return self.key == other.key

    def __cmp__(self, other):
        return cmp(self.key, other.key)

    def __lt__(self, other):
        return self.key<other.key

    def __gt__(self, other):
        return self.key>other.key

class Fp_Tree:
    '''
        1. root is none
        2. contain head table
        3. if not processed we need process data
    '''
    def __init__(self):
        self.root = Tree_Node(None)
        self.head_table = []

    def build_head_table(self, data):
        '''
        input data : 2-dimension item matrix
        :param data:
        :return:
        '''
        terms_set = set()

        cmp_by_key = lambda a, b: cmp(a.key, b.key)

        for terms in data:
            for term in terms:
                terms_set.add(term)

        head_table = []
        for key in terms_set:
            head_table.append(Head_Table_Item(key, None))

        head_table.sort(cmp_by_key)
        return head_table

    def construct(self, data):
        self.head_table = self.build_head_table(data)

        # per transaction
        for terms in data:
            cur_node = self.root                #from root node to fit in transaction record

            #data in per transaction
            for term in terms:
                tn = Tree_Node(term)
                tn.parent = cur_node
                tree_found, tree_found_idx = binary_search_left(cur_node.children, tn)

                # if data is found in the current node children vector
                #if found in the tree, there must be in the head table too
                if tree_found:
                    #modify tree part
                    target_child = cur_node.children[tree_found_idx]
                    target_child.count += 1
                    target_child.parent = cur_node
                    #modify head table
                    head_node = Head_Table_Item(term, tn)
                    head_found, head_found_idx = binary_search_left(self.head_table, head_node)
                    self.head_table[head_found_idx].count += 1
                else:
                    #not found in the children vector, may be found in the head table, so we should find it in the head
                    #table using binary search by alphabet order
                    # not found in the current tree node children array
                    i = len(cur_node.children)        #get index of the current node children vector
                    #find the position which to insert current data value, alphabet order
                    while(i>0 and cur_node.children[i-1].key>term): #current term less than i-1
                        if i > len(cur_node.children)-1:
                            cur_node.children.append(cur_node.children[i-1])
                        else:
                            cur_node.children[i] = cur_node.children[i-1]
                        i -= 1

                    if len(cur_node.children) == i:
                        cur_node.children.append(tn)
                    else:
                        cur_node.children[i] = tn

                    # after modify the tree children vector
                    # we have to modify the head table
                    item = Head_Table_Item(term, None)
                    head_found, head_found_idx = binary_search_left(self.head_table, item)
                    # not found in the tree, so there are two condition
                    # found in the head table or not found in the head table, because there are key in the table, so
                    # we just modify the next field and count field
                    if self.head_table[head_found_idx].next is None:   # found in the head table but next field is null
                        self.head_table[head_found_idx].next = tn
                        self.head_table[head_found_idx].count = tn.count
                    else:                                           # found in the head table
                        next_lnk = self.head_table[head_found_idx].next
                        while next_lnk.next is not None:
                            next_lnk = next_lnk.next
                        next_lnk.next = tn
                        self.head_table[head_found_idx].count += tn.count

                cur_node = cur_node.children[i]

    def mining(self, trans_data, trans_record, fp_result, min_support):
        '''
        for this function, make sure that find new transaction data for the new growth function
        make sure all items belong to the one transaction, namely belong to the same path in the tree
        :param trans_data: transaction data, every time is different, because transaction varies
            raw transaction data --> head node --> tree node -> expand parents data --> transaction data --> fp-growth()
        :param trans_record: also varies, record the temple transaction from tree, then store post_param in the fp_arr
        :fp_arr: store the temple and final result
        :param min_support:
        :return:
        '''
        # note every time, transaction varies, so head table is different every loop
        tree = Fp_Tree()
        tree.construct(trans_data)

        if len(tree.root.children) == 0:
            return

        #get result parts
        if trans_record is not None:
            for head_node in tree.head_table:
               if head_node.count < min_support:
                   continue
               else:
                   arr = [head_node.key]
                   for elem in trans_record:
                       arr.append(elem)
               fp_result.append((arr, head_node.count))

        #start from this line
        #traverse the tree starting with head table
        for head_node in tree.head_table:
            # trim_trans_record record the
            tree_trans_record = []
            tree_trans_record.append(head_node.key)
            if trans_record is not None:
                tree_trans_record.extend(trans_record)

            tree_node = head_node.next
            trim_trans_data = []
            while tree_node is not None:
                count = tree_node.count
                ancestors = []
                parent_node = tree_node.parent
                #parent is not root
                while parent_node is not None and parent_node.key is not None:
                    ancestors.append(parent_node.key)
                    parent_node = parent_node.parent

                while count>0:
                    count -= 1
                    trim_trans_data.append(ancestors)
                tree_node = tree_node.next

            self.mining(trim_trans_data, tree_trans_record, fp_result, min_support)



mat = read_data("data.txt")

tree = Fp_Tree()

fp_arr = []

tree.mining(mat, None, fp_arr, 3)

for fp in fp_arr:
    print fp