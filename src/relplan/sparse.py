from dataclasses import dataclass
import math
@dataclass
class CSR:
    rows:int
    cols:int
    indptr:list
    indices:list
    values:list
    def validate(self):
        if not all(isinstance(n,int) and 0<=n<=10000 for n in [self.rows,self.cols]):raise ValueError("invalid matrix shape")
        if len(self.indptr)!=self.rows+1 or not self.indptr or self.indptr[0]!=0 or self.indptr[-1]!=len(self.indices) or len(self.indices)!=len(self.values):raise ValueError("invalid CSR lengths")
        if any(not isinstance(x,int) for x in self.indptr+self.indices):raise ValueError("integer indices required")
        if any(a>b for a,b in zip(self.indptr,self.indptr[1:])) or any(not 0<=i<self.cols for i in self.indices):raise ValueError("invalid CSR indices")
        if any(not math.isfinite(v) for v in self.values):raise ValueError("finite values required")
        return self
    def row(self,i):return zip(self.indices[self.indptr[i]:self.indptr[i+1]],self.values[self.indptr[i]:self.indptr[i+1]])
    def dense(self):
        self.validate();out=[[0.0]*self.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j,v in self.row(i):out[i][j]+=v
        return out

def multiply(a,b,max_work=100000):
    a.validate();b.validate()
    if a.cols!=b.rows:raise ValueError("shape mismatch")
    ptr=[0];idx=[];values=[];work=0
    for i in range(a.rows):
        row={}
        for k,av in a.row(i):
            for j,bv in b.row(k):
                work+=1
                if work>max_work:raise ValueError("work budget exceeded")
                row[j]=row.get(j,0.0)+av*bv
        for j,v in sorted(row.items()):
            if v:idx.append(j);values.append(v)
        ptr.append(len(idx))
    return CSR(a.rows,b.cols,ptr,idx,values).validate()
