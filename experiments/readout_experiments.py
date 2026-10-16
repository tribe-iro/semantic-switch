from experiments.probe import train_probe
import argparse,time,json
from pathlib import Path
import numpy as np
from semantic_switch import Model,files,DEFAULT_JUDGE
from bench.outcomes import destination, latency, provenance, runtime
from semantic_switch.judge import scored
from semantic_switch.protocols import Question
from bench import diagnostics
from experiments.continuations import numeric
from bench import quality

def main():
    parser = argparse.ArgumentParser(description="Explicit development diagnostic; production settings remain unchanged.")
    parser.add_argument('--output', type=Path, required=True)
    output = destination(parser.parse_args().output)
    items=list(quality.check_items())[:48]
    questions=[Question('check',item['claim'],excludes=tuple(item['excludes'])) for item in items]
    states=[quality.render(item['text']) for item in items]
    groups=[item['pair'] for item in items]
    check_groups=set(sorted(set(groups))[1::2])
    labels=[0 if item['label'] else 1 for item in items]
    with runtime(Model(DEFAULT_JUDGE,device='gpu')) as rt:
     judge=rt.judge(); base=[];one=[];two=[];vec=[];numeric_rows=[];durations={k:[] for k in ('binary','lettered','counterbalanced','vectors','numeric')}
     for state,question in zip(states,questions):
      begin=time.perf_counter(); score=scored(judge,[(state,[question])])[0].scores[0];durations['binary'].append((time.perf_counter()-begin)*1000);base.append(score.probs.tolist() if not score.error else None)
      begin=time.perf_counter(); numeric_rows.append(numeric(judge,state,question));durations['numeric'].append((time.perf_counter()-begin)*1000)
      letter=Question('pick',question.text,('true','false'),question.excludes)
      begin=time.perf_counter();score=scored(judge,[(state,[letter])])[0].scores[0];durations['lettered'].append((time.perf_counter()-begin)*1000);one.append(score.probs.tolist() if not score.error else None)
      begin=time.perf_counter();score=diagnostics.counterbalanced(judge,state,question);durations['counterbalanced'].append((time.perf_counter()-begin)*1000);two.append(score.get('probs'))
      begin=time.perf_counter();features=diagnostics.vectors(rt,[(state,[question])]);durations['vectors'].append((time.perf_counter()-begin)*1000);vec.append(features['vectors'][0])
     probe=train_probe(vec,labels,groups,check_groups=check_groups)
     indices=probe['check_indices']
     def metrics(predictions,indices):
      valid=[(predictions[i],labels[i]) for i in indices if predictions[i] is not None]
      p=np.array([row[0] for row in valid]); y=np.array([row[1] for row in valid]);top=p.argmax(axis=1);confidence=p.max(axis=1)
      return {'labelled':len(indices),'valid':len(valid),'accuracy_given_prediction':float((top==y).mean()),'brier':float(((p-np.eye(2)[y])**2).sum(axis=1).mean()),'confidence_above_0_9':int((confidence>=.9).sum()),'confidence_above_0_9_correct':int(((confidence>=.9)&(top==y)).sum()),'operating_point':'unfitted confidence cutoff; no deployed mass/abstention policy'}
     data={'schema':1,'population':'development diagnostic; not an untouched generalization claim','families':len(set(groups)),'check_families':sorted(check_groups),'provenance':provenance(rt,items,boundary='explicit development diagnostic'),'numeric':{'metrics':metrics([row['probs'] for row in numeric_rows],indices),'rows':numeric_rows},'binary':metrics(base,indices),'lettered':metrics(one,indices),'counterbalanced':metrics(two,indices),'probe':{'tune':probe['tune'],'check':probe['check'],'correct':sum(int(np.argmax(p))==labels[i] for p,i in zip(probe['probs'],indices))},'latency':{name:latency(values) for name,values in durations.items()},'production_promoted':False,'missing_evidence':['untouched task families','fixed held-out acceptance calibration','resident E2B/E4B paired residual comparison','compatible externally trained activated-LoRA weights']}
     files.publish(output,data)
     print(json.dumps({k:data[k] for k in ('binary','lettered','counterbalanced','probe','latency')}),flush=True)


if __name__ == "__main__":
    main()
