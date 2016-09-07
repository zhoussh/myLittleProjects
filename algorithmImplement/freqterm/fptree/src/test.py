from collections import defaultdict


#open file
with open("data.txt", 'r') as f_r:
    mat = []
    for line in f_r:
        line = line.strip().decode("utf-8")
        parts = line.split(',')
        mat.append(parts)
    print mat

# head table

term_count = defaultdict(int)
cmp_by_node_val = lambda a,b: cmp(a.val, b.val)
