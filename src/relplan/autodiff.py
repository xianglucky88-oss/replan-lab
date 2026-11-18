import math
class Value:
    def __init__(self,value,parents=()):
        self._value=float(value);self.parents=parents;self.grad=0.0
        if not math.isfinite(self.value):raise ValueError("finite values required")
    @property
    def value(self):return self._value
    @staticmethod
    def lift(v):return v if isinstance(v,Value) else Value(v)
    def __add__(self,other):
        o=self.lift(other);return Value(self.value+o.value,((self,1.0),(o,1.0)))
    __radd__=__add__
    def __mul__(self,other):
        o=self.lift(other);return Value(self.value*o.value,((self,o.value),(o,self.value)))
    __rmul__=__mul__
    def sin(self):return Value(math.sin(self.value),((self,math.cos(self.value)),))
    def exp(self):
        v=math.exp(self.value);return Value(v,((self,v),))
    def backward(self,max_nodes=10000):
        order=[];seen=set();stack=[(self,False)]
        while stack:
            node,done=stack.pop()
            if done:order.append(node);continue
            if id(node) in seen:continue
            seen.add(id(node))
            if len(seen)>max_nodes:raise ValueError("tape budget exceeded")
            stack.append((node,True))
            stack.extend((p,False) for p,_ in node.parents)
        for node in order:node.grad=0.0
        self.grad=1.0
        for node in reversed(order):
            for parent,weight in node.parents:parent.grad+=node.grad*weight
        return self.value

def evaluate(x,y):
    a=Value(x);b=Value(y);result=(a*b+a.sin())*a
    result.backward()
    return {"value":result.value,"dx":a.grad,"dy":b.grad}
