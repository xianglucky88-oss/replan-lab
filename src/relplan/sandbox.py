import wasmtime

def invoke(wat,args=(),fuel=100000,memory_bytes=1048576,export="run"):
    if not isinstance(wat,str) or len(wat)>100000:raise ValueError("module size limit")
    if not isinstance(fuel,int) or not 1<=fuel<=10_000_000:raise ValueError("invalid execution budget")
    if not isinstance(memory_bytes,int) or not 65536<=memory_bytes<=16_777_216:raise ValueError("invalid memory limit")
    config=wasmtime.Config();config.consume_fuel=True
    engine=wasmtime.Engine(config);store=wasmtime.Store(engine)
    store.set_fuel(fuel);store.set_limits(memory_size=memory_bytes,instances=1,memories=1,tables=1)
    module=wasmtime.Module(engine,wat)
    if module.imports:raise ValueError("host imports are disabled")
    instance=wasmtime.Instance(store,module,[])
    function=instance.exports(store).get(export)
    if not isinstance(function,wasmtime.Func):raise ValueError("missing function export")
    if any(not isinstance(v,int) for v in args):raise ValueError("integer ABI required")
    value=function(store,*args)
    return {"value":value,"fuel_remaining":store.get_fuel()}
