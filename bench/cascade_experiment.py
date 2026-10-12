import time,json
from pathlib import Path
from semantic_switch import Model, Check, files, DEFAULT_JUDGE, PRECISE_JUDGE
from bench.outcomes import runtime
from bench import quality
from semantic_switch.inline import ask_many
from semantic_switch.compiler import digest
from bench import diagnostics

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Explicit development diagnostic; production settings remain unchanged.")
    parser.parse_args()
    items=list(quality.check_items());groups=[]
    for item in items:
     fact,index=item['pair'].rsplit(':',1); pairs=len(quality.CHECKS[fact][2]);groups.append(f'{fact}:{int(index)%(pairs//2)}')
    tune=[int(digest(['cascade-split',g])[:8],16)%2==0 for g in groups]
    records=[]
    with runtime(Model(DEFAULT_JUDGE,device='gpu'),precise=Model(PRECISE_JUDGE,device='gpu')) as rt:
     loads={}
     for role in ('judge','precise'):
      begin=time.perf_counter();rt.judge(role);loads[role]=(time.perf_counter()-begin)*1000
     outputs={}
     for role in ('judge','precise'):
      results=[None]*len(items); durations=[]
      for fact in quality.CHECKS:
       indices=[i for i,item in enumerate(items) if item['fact']==fact]; first=items[indices[0]];spec=Check(first['claim'],tuple(first['excludes']))
       for start in range(0,len(indices),8):
        window=indices[start:start+8];begin=time.perf_counter();out=ask_many(lambda:rt.judge(role),[quality.render(items[i]['text']) for i in window],{'check':spec});elapsed=(time.perf_counter()-begin)*1000
        durations.append({'inputs':len(window),'batch_ms':elapsed})
        for i,row in zip(window,out): results[i]=row['check']
      outputs[role]=results
      files.publish(Path('bench/results/cascade-experiment-v026.json'),{'partial':True,'completed':role,'loads_ms':loads,'rows':[{**item,'family':groups[i],'split':'tune' if tune[i] else 'check','output':results[i].as_dict(True)} for i,item in enumerate(items)],'durations':durations})
     def accepted(result,theta):
      if result.p_true is None or result.mass is not None and result.mass<.5 or result.status=='failed': return None
      return True if result.p_true>=theta else False if 1-result.p_true>=theta else None
     def choose(role,indices):
      best,count=None,0
      from semantic_switch.tools.policy import choose as choose_policy
      scores = []
      for i in indices:
       result = outputs[role][i]
       valid = result.p_true is not None and (result.mass is None or result.mass >= .5)
       confidence = max(result.p_true, 1-result.p_true) if valid else -1.
       scores.append((confidence, (result.p_true >= .5) == items[i]['label'] if valid else False))
      enough,best=choose_policy(scores,.95)
      count=sum(confidence >= best for confidence,_ in scores) if best is not None else 0
      return {'theta':best if best is not None else .9,'state':'met-on-tune' if best is not None else 'insufficient-or-missed','accepted':count}
     tune_indices=[i for i,train in enumerate(tune) if train];check_indices=[i for i,train in enumerate(tune) if not train]
     small=choose('judge',tune_indices);large=choose('precise',tune_indices)
     residue_tune=[i for i in tune_indices if accepted(outputs['judge'][i],small['theta']) is None];residue_policy=choose('precise',residue_tune)
     residue_check=[i for i in check_indices if accepted(outputs['judge'][i],small['theta']) is None]
     def metrics(predict,indices):
      got=[predict(i) for i in indices];settled=[value==items[i]['label'] for i,value in zip(indices,got) if value is not None]
      return {'labelled':len(indices),'accepted':len(settled),'correct':sum(settled),'wrong':len(settled)-sum(settled),'coverage':len(settled)/len(indices) if indices else None,'precision':sum(settled)/len(settled) if settled else None}
     report={'schema':1,'population':'existing development probe; grouped tune/check experiment; not untouched task evidence','production_promoted':False,'provenance':diagnostics.provenance(rt,items),'identities':{role:rt.identity(role) for role in outputs},'loads_ms':loads,'policies':{'small':small,'large':large,'residue':residue_policy},'small':metrics(lambda i:accepted(outputs['judge'][i],small['theta']),check_indices),'large':metrics(lambda i:accepted(outputs['precise'][i],large['theta']),check_indices),'residue':metrics(lambda i:accepted(outputs['precise'][i],residue_policy['theta']),residue_check),'cascade':metrics(lambda i:accepted(outputs['judge'][i],small['theta']) if i not in residue_check else accepted(outputs['precise'][i],residue_policy['theta']),check_indices),'fixed_band_0.9': {'small':metrics(lambda i:accepted(outputs['judge'][i],.9),check_indices),'residue':metrics(lambda i:accepted(outputs['precise'][i],.9),[i for i in check_indices if accepted(outputs['judge'][i],.9) is None]),'cascade':metrics(lambda i:accepted(outputs['judge'][i],.9) if accepted(outputs['judge'][i],.9) is not None else accepted(outputs['precise'][i],.9),check_indices)},'rows':[{**item,'family':groups[i],'split':'tune' if tune[i] else 'check','small':outputs['judge'][i].as_dict(True),'large':outputs['precise'][i].as_dict(True)} for i,item in enumerate(items)]}
     files.publish(Path('bench/results/cascade-experiment-v026.json'),report)
     print(json.dumps({key:report[key] for key in ('policies','small','large','residue','cascade')}),flush=True)


if __name__ == "__main__":
    main()
