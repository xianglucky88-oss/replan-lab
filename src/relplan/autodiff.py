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

class ExpressionError(ValueError):
    """A scalar expression cannot be evaluated in its domain."""

def evaluate_terms(compiled,inputs,weights):
    """Evaluate ``sum_i w_i * f_i`` and differentiate with respect to every weight.

    ``compiled`` is a list of term descriptors in evaluation order:

    ``("const",)``
        Term value is ``w_i`` itself (intercept).
    ``("var", field)``
        Term value is ``w_i * inputs[field]``.
    ``("mul"|"sin"|"exp", position)``
        Term value is ``w_i * op(term_values[position])`` for an earlier term.

    A term may be referenced by any number of later terms.  ``Value.backward``
    accumulates into shared intermediates, so every reference contributes and
    repeated references are never dropped.  ``gradients[i]`` is d(value)/dw_i.
    The caller receives the exact same derivative meaning it would get by
    building the equivalent graph with ``Value`` itself.
    """
    if not isinstance(compiled,(list,tuple)):raise ExpressionError("compiled terms must be a sequence")
    weights=list(weights)
    if len(weights)!=len(compiled):raise ExpressionError("weight vector must align with terms")
    weight_nodes=[]
    for index,weight in enumerate(weights):
        if isinstance(weight,bool) or not isinstance(weight,(int,float)):raise ExpressionError("weights must be real numbers")
        try:weight_nodes.append(Value(weight))
        except ValueError as error:raise ExpressionError(str(error)) from error
    values=[]
    for index,term in enumerate(compiled):
        if not isinstance(term,(list,tuple)) or not term:raise ExpressionError("invalid compiled term")
        kind=term[0];node=weight_nodes[index]
        if kind=="const":
            value=node
        elif kind=="var":
            if len(term)!=2 or not isinstance(term[1],str):raise ExpressionError("invalid variable term")
            field=term[1]
            if field not in inputs:raise ExpressionError(f"missing input {field!r}")
            try:value=node*Value(inputs[field])
            except ValueError as error:raise ExpressionError(str(error)) from error
        elif kind in {"mul","mul2","sin","exp"}:
            if len(term)!=2 or not isinstance(term[1],int) or not 0<=term[1]<index:raise ExpressionError("term references must point backwards")
            base=values[term[1]]
            try:
                if kind=="mul":value=node*base
                elif kind=="mul2":value=node*base*base
                elif kind=="sin":value=node*base.sin()
                else:value=node*base.exp()
            except (ValueError,OverflowError) as error:raise ExpressionError(str(error)) from error
        else:raise ExpressionError("unsupported compiled term")
        values.append(value)
    total=None
    for value in values:total=value if total is None else total+value
    if total is None:raise ExpressionError("empty expression")
    total.backward()
    return {"value":total.value,"gradients":[node.grad for node in weight_nodes],"term_values":[value.value for value in values]}
