import time,tracemalloc,json,tempfile
from pathlib import Path
import numpy as np
from semantic_switch import Runtime, Model, DEFAULT_JUDGE, files
from bench import diagnostics
from bench.outcomes import runtime
from semantic_switch.heads import LexicalHead,features
from semantic_switch.judge import scored,CONFORMANCE_PROBES,CONFORMANCE_QUESTIONS

def main():
    import argparse
    argparse.ArgumentParser(description="Explicit primitive/native controls.").parse_args()
    output={}
    def measurement(fn,n=500):
     samples=[]
     for _ in range(n):
      start=time.perf_counter();fn();samples.append((time.perf_counter()-start)*1e6)
     tracemalloc.start()
     for _ in range(100):fn()
     _,peak=tracemalloc.get_traced_memory();tracemalloc.stop()
     return {'samples':n,'p50_us':float(np.percentile(samples,50)),'p95_us':float(np.percentile(samples,95)),'python_peak_bytes':peak,'timing_includes_tracemalloc':False,'allocation_samples':100}
    text='A customer requests a refund for a duplicate charge. '*10
    keys=np.array(sorted(features(text)),dtype=np.int32);rng=np.random.default_rng(42)
    head=LexicalHead(('true','false'),keys,rng.normal(0,.1,(len(keys),2)).astype(np.float32),np.zeros(2,dtype=np.float32))
    output['lexical_complete']=measurement(lambda:head.probs(text))
    features_=features(text)
    output['lexical_reduction']=measurement(lambda:head.probs_features(features_))
    def original_loop():
     z=head.b.astype(np.float64,copy=True)
     for key,value in features_.items():
      row=head.rows.get(key)
      if row is not None:z+=value*head.w[row]
     e=np.exp(z-z.max());return e/e.sum()
    output['lexical_original_loop']=measurement(original_loop)
    output['probability_difference']=float(np.abs(original_loop()-head.probs_features(features_)).max())
    with tempfile.TemporaryDirectory() as tmp:
     path=Path(tmp)/'exact.yaml';files.publish(path,{'name':'exact','fields':{'enabled':{'type':'bool'}},'facts':{},'cases':[{'id':'on','when':'enabled','then':'on'}],'otherwise':'off'})
     with Runtime() as rt:
      switch=rt.load(path);output['exact_decision']=measurement(lambda:switch({'enabled':True}))
      assert not rt.info()['loaded']
    with runtime(Model(DEFAULT_JUDGE,device='gpu')) as rt:
     judge=rt.judge();items=[(CONFORMANCE_PROBES[0],CONFORMANCE_QUESTIONS[:1])]
     expected=scored(judge,items)[0].scores[0].probs
     zero=diagnostics.activated_suffix(rt,items)
     after=scored(judge,items)[0].scores[0].probs
     output['zero_adapter_control']={'max_diff':float(np.abs(expected-np.exp(zero['logp'][0])).max()),'after_diff':float(np.abs(expected-after).max()),'errors':zero['errors'],'nonzero_adapter_validated':False}
     output['provenance']=diagnostics.provenance(rt,items)
     files.publish(Path('bench/results/primitives-v026.json'),output)
     print(json.dumps({key:value for key,value in output.items() if key!='provenance'}),flush=True)


if __name__ == "__main__":
    main()
