from . import relations,sandbox,sparse,autodiff,randomstream

def _nonneg_int(d, key, default):
    value = d.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(key + " must be a non-negative integer")
    return value


def relation_call(d):
    search_budget = _nonneg_int(d, "search_budget", relations.DEFAULT_SEARCH_BUDGET)
    execution_budget = _nonneg_int(d, "execution_budget", relations.DEFAULT_EXECUTION_BUDGET)
    return relations.run_query(d["plan"], d["tables"],
                               search_budget=search_budget,
                               execution_budget=execution_budget)

def sparse_call(d):
    def matrix(r):return sparse.CSR(r['rows'],r['cols'],r['indptr'],r['indices'],r['values'])
    result=sparse.multiply(matrix(d['left']),matrix(d['right']));return {'rows':result.rows,'cols':result.cols,'indptr':result.indptr,'indices':result.indices,'values':result.values}
def stream_call(d):
    if not 0<=d.get('count',5)<=1000:raise ValueError('sample limit')
    stream=randomstream.RandomStream(d.get('seed',42));values=[stream.uniform() for _ in range(d.get('count',5))]
    return {'values':values,'state':stream.dump()}
OPERATIONS={'relations':relation_call,'sandbox':lambda d:sandbox.invoke(d['wat'],d.get('args',[]),d.get('fuel',100000)),
 'sparse':sparse_call,'autodiff':lambda d:autodiff.evaluate(d['x'],d['y']),'randomstream':stream_call}
EXAMPLES={
 'relations':{'tables':{'txn':[{'id':1,'cat':'a'},{'id':2,'cat':'b'},{'id':3,'cat':None},{'id':4,'cat':'a'}],
  'detail':[{'txn_id':1,'amt':10},{'txn_id':1,'amt':10},{'txn_id':2,'amt':5}],'dim_cat':[]},
  'plan':{'op':'project','columns':['txn.id','detail.amt','dim_cat.name'],'input':
   {'op':'join','kind':'left','predicate':['eq',['col','txn.cat'],['col','dim_cat.code']],
    'left':{'op':'join','kind':'left','predicate':['eq',['col','txn.id'],['col','detail.txn_id']],
     'left':{'op':'scan','table':'txn'},'right':{'op':'scan','table':'detail'}},
    'right':{'op':'scan','table':'dim_cat'}}},'search_budget':256,'execution_budget':100000},
 'sandbox':{'wat':'(module (func (export "run") (param i64 i64) (result i64) local.get 0 local.get 1 i64.add))','args':[2,5]},
 'sparse':{'left':{'rows':1,'cols':2,'indptr':[0,2],'indices':[0,1],'values':[2,3]},'right':{'rows':2,'cols':1,'indptr':[0,1,2],'indices':[0,0],'values':[4,5]}},
 'autodiff':{'x':2,'y':3},'randomstream':{'seed':42,'count':5},
}
