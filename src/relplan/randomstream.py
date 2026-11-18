import hashlib,json,math
class RandomStream:
    def __init__(self,seed,domain="root",counter=0):
        if not isinstance(seed,int) or seed<0 or seed>=2**128 or not isinstance(counter,int) or not 0<=counter<2**128:raise ValueError("invalid stream position")
        if not isinstance(domain,str) or len(domain.encode())>4096:raise ValueError("domain limit")
        self.seed=seed;self.domain=domain;self.counter=counter;self.spare=None;self.children=set()
    def uint64(self):
        if self.counter>=2**128-1:raise ValueError("counter exhausted")
        label=self.domain.encode();data=self.seed.to_bytes(16,"big")+len(label).to_bytes(4,"big")+label+self.counter.to_bytes(16,"big")
        self.counter+=1
        return int.from_bytes(hashlib.sha256(data).digest()[:8],"big")
    def uniform(self):return (self.uint64()>>11)/2**53
    def normal(self):
        if self.spare is not None:v=self.spare;self.spare=None;return v
        u=1.0-self.uniform();v=self.uniform();r=math.sqrt(-2*math.log(u));theta=2*math.pi*v
        self.spare=r*math.sin(theta);return r*math.cos(theta)
    def split(self,label):
        if not isinstance(label,str) or not label or label in self.children:raise ValueError("duplicate or empty split label")
        self.children.add(label)
        # Length-delimited path prevents names containing separators from aliasing.
        return RandomStream(self.seed,self.domain+"/"+str(len(label.encode()))+":"+label)
    def jump(self,n):
        if not isinstance(n,int) or n<0 or self.counter+n>=2**128:raise ValueError("invalid jump")
        self.counter+=n;self.spare=None
    def dump(self):return json.dumps({"version":1,"seed":str(self.seed),"domain":self.domain,"counter":str(self.counter),"spare":self.spare,"children":sorted(self.children)},sort_keys=True)
    @classmethod
    def load(cls,text):
        d=json.loads(text)
        if d.get("version")!=1:raise ValueError("unsupported stream version")
        s=cls(int(d["seed"]),d["domain"],int(d["counter"]));s.spare=d["spare"]
        if s.spare is not None and not math.isfinite(s.spare):raise ValueError("invalid transform cache")
        children=d['children']
        if not isinstance(children,list) or len(children)!=len(set(children)) or any(not isinstance(x,str) or not x for x in children):raise ValueError('invalid split registry')
        s.children=set(children);return s
