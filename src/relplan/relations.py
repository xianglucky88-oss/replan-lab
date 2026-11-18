from copy import deepcopy

def predicate(p,row):
    if not isinstance(p,list) or not p:raise ValueError("invalid predicate")
    op=p[0]
    if op=="col":return row.get(p[1])
    if op=="lit":return p[1]
    if op=="and":
        a,b=predicate(p[1],row),predicate(p[2],row)
        return False if a is False or b is False else None if a is None or b is None else bool(a and b)
    if op not in {"eq","lt","gt"}:raise ValueError("unsupported predicate")
    a,b=predicate(p[1],row),predicate(p[2],row)
    if a is None or b is None:return None
    return a==b if op=="eq" else a<b if op=="lt" else a>b

def execute(plan,tables,budget=100000):
    steps=[0]
    def visit(p,depth=0):
        if depth>30:raise ValueError("plan depth limit")
        op=p["op"]
        if op=="scan":
            name=p["table"]
            if name not in tables:raise ValueError("unknown table")
            rows=[{name+"."+k:v for k,v in r.items()} for r in tables[name]]
        elif op=="filter":rows=[r for r in visit(p["input"],depth+1) if predicate(p["predicate"],r) is True]
        elif op=="project":rows=[{k:r.get(k) for k in p["columns"]} for r in visit(p["input"],depth+1)]
        elif op=="join":
            if p.get("kind","inner") not in {"inner","left"}:raise ValueError("unsupported join")
            left=visit(p["left"],depth+1);right=visit(p["right"],depth+1);rows=[]
            if len(left)*max(1,len(right))+steps[0]>budget:raise ValueError("execution budget exceeded")
            right_keys=set().union(*(r.keys() for r in right)) if right else set(p.get("right_columns",[]))
            for l in left:
                hits=[dict(l,**r) for r in right if predicate(p["predicate"],dict(l,**r)) is True]
                rows.extend(hits)
                if not hits and p.get("kind")=="left":rows.append(dict(l,**{k:None for k in right_keys}))
        else:raise ValueError("unsupported relational operator")
        steps[0]+=len(rows)
        if steps[0]>budget:raise ValueError("execution budget exceeded")
        return rows
    return visit(plan)

def columns(pred):
    if pred[0]=="col":return {pred[1]}
    if pred[0]=="lit":return set()
    return set().union(*(columns(x) for x in pred[1:] if isinstance(x,list)))
def optimize(plan,limit=256):
    p=deepcopy(plan);seen=[0]
    def visit(x):
        seen[0]+=1
        if seen[0]>limit:raise ValueError("optimization budget exceeded")
        for k in ["input","left","right"]:
            if k in x:x[k]=visit(x[k])
        if x["op"]=="filter" and x["input"]["op"]=="join":
            j=x["input"];l=j["left"]
            if j.get("kind","inner")=="inner" and l["op"]=="scan" and columns(x["predicate"]) and all(c.startswith(l["table"]+".") for c in columns(x["predicate"])):
                j["left"]={"op":"filter","predicate":x["predicate"],"input":l};return j
        return x
    return visit(p)
