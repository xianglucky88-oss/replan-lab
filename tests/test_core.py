import math,random,collections
import pytest
import wasmtime
from relplan import relations,sandbox,sparse,autodiff,randomstream

def test_optimizer_preserves_bags_and_nulls():
    tables={'a':[{'id':1},{'id':1},{'id':None},{'id':2}],'b':[{'id':1},{'id':1},{'id':None}]}
    join={'op':'join','left':{'op':'scan','table':'a'},'right':{'op':'scan','table':'b'},'predicate':['eq',['col','a.id'],['col','b.id']]}
    plan={'op':'filter','input':join,'predicate':['gt',['col','a.id'],['lit',0]]}
    old=relations.execute(plan,tables);new=relations.execute(relations.optimize(plan),tables)
    assert old==new and len(new)==4
    join['kind']='left'
    assert relations.optimize(plan)['op']=='filter'
    assert relations.execute(relations.optimize(plan),tables)==relations.execute(plan,tables)
def test_sql_null_and_budgets():
    assert relations.predicate(['eq',['lit',None],['lit',None]],{}) is None
    with pytest.raises(ValueError):relations.execute({'op':'scan','table':'a'},{'a':[{},{}]},budget=1)
def test_actual_wasm_and_resource_rejection():
    wat='(module (func (export "run") (param i64) (result i64) local.get 0 i64.const 2 i64.mul))'
    assert sandbox.invoke(wat,[7])['value']==14
    with pytest.raises(wasmtime.Trap):sandbox.invoke('(module (func (export "run") (loop br 0)))',[],fuel=100)
    with pytest.raises(ValueError):sandbox.invoke('(module (import "env" "x" (func)))')
    with pytest.raises(Exception):sandbox.invoke('(module (memory 100) (func (export "run")))')
def test_sparse_duplicate_entries_against_dense():
    a=sparse.CSR(2,2,[0,3,4],[1,0,1,0],[2,3,-1,4]);b=sparse.CSR(2,2,[0,1,3],[0,0,1],[5,6,7])
    aa,bb=a.dense(),b.dense();expected=[[sum(aa[i][k]*bb[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
    assert sparse.multiply(a,b).dense()==expected
    with pytest.raises(ValueError):sparse.multiply(a,b,max_work=1)
    with pytest.raises(ValueError):sparse.CSR(1,2,[0,1],[3],[1]).validate()
@pytest.mark.parametrize('x,y',[(2.,3.),(-.7,4.),(0.,1.),(10.,-.1)])
def test_autodiff_independent_derivative(x,y):
    r=autodiff.evaluate(x,y)
    assert r['value']==pytest.approx(x*x*y+x*math.sin(x))
    assert r['dx']==pytest.approx(2*x*y+math.sin(x)+x*math.cos(x))
    assert r['dy']==pytest.approx(x*x)
def test_shared_autodiff_node_and_reset():
    x=autodiff.Value(3);y=x*x+x;x0=y.backward();assert x0==12 and x.grad==7
    y.backward();assert x.grad==7
def test_random_stream_split_restore_and_cached_normal():
    s=randomstream.RandomStream(42);child=s.split('a/b');other=s.split('a')
    assert child.uniform()!=other.split('b').uniform()
    with pytest.raises(ValueError):s.split('a/b')
    s.normal();restored=randomstream.RandomStream.load(s.dump())
    assert [s.normal() for _ in range(7)]==[restored.normal() for _ in range(7)]
    a,b=randomstream.RandomStream(8),randomstream.RandomStream(8)
    for _ in range(4):a.uint64()
    b.jump(4);assert a.uint64()==b.uint64()
