import numpy as np

class s2t(object):
    
    def __init__(self, filename):
        self.t2v = {}
        self.dim = None
        for line in open(filename):
            line = line.strip().split('\t')
            t = line[0]
            v = np.array([float(x) for x in line[1].split()])
            if self.dim is None:
                self.dim = len(v)
            else:
                v = v[:self.dim]
            self.t2v[t] = v
    
    def embed(self, seq):
        if seq.find(' ') > 0:
            s = seq.strip().split()
        else:
            s = list(seq.strip())
        rst = []
        for x in s:
            v = self.t2v.get(x)
            if v is None:
                continue
            rst.append(v)
        return np.array(rst)
    
    def embed_normalized(self, seq, length=50):
        rst = self.embed(seq)
        if len(rst) > length:
            return rst[:length]
        elif len(rst) < length:
            return np.concatenate((rst, np.zeros((length - len(rst), self.dim))))
        return rst