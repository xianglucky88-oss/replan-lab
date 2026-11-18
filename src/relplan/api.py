from . import relations,sandbox,sparse,autodiff,randomstream

def relation_call(d):
    plan=relations.optimize(d['plan']);return {'plan':plan,'rows':relations.execute(plan,d['tables'])}
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
 'relations':{'tables':{'a':[{'x':1},{'x':None},{'x':3}]},'plan':{'op':'filter','predicate':['gt',['col','a.x'],['lit',1]],'input':{'op':'scan','table':'a'}}},
 'sandbox':{'wat':'(module (func (export "run") (param i64 i64) (result i64) local.get 0 local.get 1 i64.add))','args':[2,5]},
 'sparse':{'left':{'rows':1,'cols':2,'indptr':[0,2],'indices':[0,1],'values':[2,3]},'right':{'rows':2,'cols':1,'indptr':[0,1,2],'indices':[0,0],'values':[4,5]}},
 'autodiff':{'x':2,'y':3},'randomstream':{'seed':42,'count':5},
}
