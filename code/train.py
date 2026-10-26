"""Reproduce the frozen final recipe from supplied train data, on CPU.

No previous checkpoint, teacher cache, validation labels or test inputs are read.
Prediction uses submission.py with NumPy; training additionally requires Torch
and an XGBoost interpreter (optionally supplied through --tree-python).
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '4'
os.environ['PYTHONIOENCODING'] = 'utf-8'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import numpy as np
from common import DATA, PROTOCOL, load_inputs, read_jsonl, parse_answer
from submission import (answer_groups, answer_types, standardize, numpy_logits,
    tree_features, tree_scores, prepare_inputs, normalize, pool_scores)

HERE = Path(__file__).resolve().parent
def read(path):
    return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export_trees(booster, rounds, depth):
    nodes = {key:[] for key in ('feature','threshold','left','right','missing','leaf')}
    roots = []
    for text in booster.get_dump(dump_format='json')[:rounds]:
        entries = {}
        def collect(node):
            entries[node['nodeid']] = node
            for child in node.get('children',[]):collect(child)
        collect(json.loads(text))
        offsets = {key:len(nodes['feature'])+i for i,key in enumerate(sorted(entries))}
        roots.append(offsets[0])
        for key in sorted(entries):
            node = entries[key];leaf = 'leaf' in node
            nodes['feature'].append(-1 if leaf else int(node['split'][1:]))
            nodes['threshold'].append(0 if leaf else node['split_condition'])
            nodes['leaf'].append(node.get('leaf',0))
            for field,branch in (('left','yes'),('right','no'),('missing','missing')):
                nodes[field].append(offsets[key] if leaf else offsets[node[branch]])
    result = {f'tree_{key}':np.asarray(values,np.float32 if key in ('threshold','leaf') else np.int32)
              for key,values in nodes.items()}
    result.update(tree_roots=np.asarray(roots,np.int32),tree_max_depth=np.asarray(depth,np.int32))
    return result


def tree_worker(job_path, output, settings):
    # This branch deliberately imports no Torch: an existing second interpreter
    # with XGBoost may be used without installing Torch into that environment.
    import xgboost as xgb
    with np.load(job_path,allow_pickle=False) as saved:
        features,labels,margin = (saved[k] for k in ('features','labels','margin'))
    data = xgb.DMatrix(features.reshape(-1,features.shape[-1]),label=labels.ravel(),
        group=np.full(len(features),4),base_margin=margin.ravel(),nthread=4)
    params = dict(objective='rank:pairwise',tree_method='hist',max_bin=128,eta=.03,
        reg_lambda=20,min_child_weight=10,subsample=.8,colsample_bytree=.8,
        max_depth=settings['tree_depth'],seed=42,nthread=4,disable_default_eval_metric=1,device='cpu')
    booster = xgb.train(params,data,num_boost_round=settings['tree_rounds'])
    weights = export_trees(booster,settings['tree_rounds'],settings['tree_depth'])
    np.testing.assert_allclose(tree_scores(features,weights,margin),
        booster.predict(data,output_margin=True).reshape(-1,4),rtol=1e-5,atol=3e-5)
    np.savez(output,**weights)
    write(output.with_suffix('.json'),dict(xgboost=xgb.__version__,numpy=np.__version__,python=sys.version))


def choose_tree_python(explicit):
    candidates = [explicit] if explicit else [Path(sys.executable),Path(sys.executable).parents[2]/'python.exe']
    for path in candidates:
        if path is None or not path.is_file():continue
        result = subprocess.run([str(path),'-B','-c','import xgboost;print(xgboost.__version__)'],
            capture_output=True,text=True,env=os.environ.copy())
        if result.returncode==0:return path.resolve()
    raise RuntimeError('Install requirements-training.txt, or pass --tree-python PATH_TO_PYTHON_WITH_XGBOOST.')


def labels_for(bank, references):
    if len(bank)!=len(references):raise ValueError('Training references are not aligned.')
    labels=[]
    for row,ref in zip(bank,references):
        if row['id']!=ref['id'] or row['question']!=ref['question']:raise ValueError('Reference mismatch.')
        gold=parse_answer(ref['answer'])
        answers=[parse_answer(c['text']) for c in row['candidates']]
        if gold is None or gold not in answers:raise ValueError('Missing correct candidate.')
        labels.append([a is not None and a==gold for a in answers])
    return np.asarray(labels,np.float32)


def make_folds(strata, count, seed):
    rng=np.random.default_rng(seed);folds=[[] for _ in range(count)]
    for value in np.unique(strata):
        indices=np.flatnonzero(strata==value);rng.shuffle(indices)
        for fold,part in zip(folds,np.array_split(indices,count)):fold.extend(part.tolist())
    return [np.sort(fold) for fold in folds]


def fit_base(bank, embeddings, labels, indices, settings, directory, interpreter, config_path):
    import torch
    from torch import nn
    from torch.nn import functional as F
    directory.mkdir()
    subset=[bank[i] for i in indices]
    flat=embeddings[indices].reshape(-1,embeddings.shape[-1])
    mean=flat.mean(0,dtype=np.float64).astype(np.float32)
    scale=np.maximum(flat.std(0,dtype=np.float64),1e-6).astype(np.float32)
    x=standardize(embeddings[indices],mean,scale,settings['input_clip'])
    tx=torch.from_numpy(x);target=torch.from_numpy(labels[indices])
    valid=torch.from_numpy(np.asarray([[parse_answer(c['text']) is not None for c in row['candidates']] for row in subset]))
    torch.manual_seed(settings['seed'])
    network=nn.Linear(x.shape[-1],1);nn.init.zeros_(network.weight);nn.init.zeros_(network.bias)
    optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    generator=torch.Generator().manual_seed(settings['seed'])
    for epoch in range(1,settings['epochs']+1):
        losses=[]
        for batch in torch.randperm(len(tx),generator=generator).split(settings['batch_size']):
            optimizer.zero_grad(set_to_none=True)
            scores=network(tx[batch]).squeeze(-1);y=target[batch];ok=valid[batch]
            bce=(F.binary_cross_entropy_with_logits(scores,y,reduction='none')*ok).sum()/ok.sum().clamp_min(1)
            pairs=(y.unsqueeze(2)>y.unsqueeze(1))&ok.unsqueeze(2)&ok.unsqueeze(1)
            counts=pairs.sum(dim=(1,2))
            each=(F.softplus(scores.unsqueeze(1)-scores.unsqueeze(2))*pairs).sum(dim=(1,2))/counts.clamp_min(1)
            ranking=each[counts>0].mean() if (counts>0).any() else scores.sum()*0
            loss=ranking+settings['aux_bce_weight']*bce
            loss.backward();nn.utils.clip_grad_norm_(network.parameters(),settings['gradient_clip']);optimizer.step()
            losses.append(float(loss.detach()))
        print(f'[{directory.name} linear {epoch}/{settings["epochs"]}] loss={np.mean(losses):.6f}',flush=True)
    base=dict(mean=mean,scale=scale,linear_weight=network.weight.detach().numpy()[0].copy(),
              linear_bias=network.bias.detach().numpy()[0].copy())
    relative=(x-x.mean(1,keepdims=True)).reshape(-1,x.shape[-1])
    values,vectors=np.linalg.eigh(relative.T@relative/len(relative))
    order=np.argsort(values)[::-1][:settings['projection_dimension']]
    base['projection']=(vectors[:,order]/np.sqrt(np.maximum(values[order],1e-6))).astype(np.float32)
    kinds=answer_types(subset);ok=valid.numpy();counts=np.zeros((2,4),np.int64)
    for correct in (0,1):counts[correct]=np.bincount(kinds[ok&(labels[indices].astype(bool)==bool(correct))],minlength=4)
    probability=(counts+1)/(counts.sum(axis=1,keepdims=True)+4)
    base['answer_type_log_ratio']=np.log(probability[1]/probability[0]).astype(np.float32)
    margin=numpy_logits(x,base).astype(np.float32)
    job=directory/'tree-job.npz'
    np.savez(job,features=tree_features(subset,x,base),labels=labels[indices],margin=margin)
    subprocess.run([str(interpreter),'-B',str(Path(__file__).resolve()),'--config',str(config_path),
        '--tree-worker',str(job),'--tree-output',str(directory/'trees.npz')],check=True,cwd=HERE,env=os.environ.copy())
    with np.load(directory/'trees.npz',allow_pickle=False) as saved:trees={k:saved[k].copy() for k in saved.files}
    job.unlink()  # Only the exact temporary file created in this function.
    # Match the frozen model's key order as well as array values.
    base={**{k:base[k] for k in ('mean','scale','linear_weight','linear_bias')},**trees,
          'answer_type_log_ratio':base['answer_type_log_ratio'],'projection':base['projection']}
    config=dict(format='new_ranker_trees_v1',protocol=PROTOCOL,input_clip=settings['input_clip'],
        fit_split='train',selection='validation',arithmetic_checks=False,feature_mode='projected',
        base_margin_scale=1.,answer_prior_strength=settings['answer_prior_strength'],
        selected_rounds=settings['tree_rounds'],projection_dimension=settings['projection_dimension'],hard_training_weight=1.)
    np.savez(directory/'model.npz',**base);write(directory/'config.json',config)
    write(directory/'fit-indices.json',indices.tolist())
    return base,config


def fit_heads(data, gold, settings, directory):
    import torch
    from torch import nn
    from torch.nn import functional as F
    features,scores,ids,mask=data
    hard=mask.sum(1)>=2;values=features[hard][ids[hard]>=0]
    shared=dict(feature_mean=values.mean(0,dtype=np.float64).astype(np.float32),
                feature_scale=np.maximum(values.std(0,dtype=np.float64),.05).astype(np.float32))
    x=torch.from_numpy(normalize(features[hard],shared,settings['feature_clip']))
    reference=torch.from_numpy(pool_scores(scores[hard],ids[hard],'max')/settings['reference_temperature'])
    tid,tmask=torch.from_numpy(ids[hard]),torch.from_numpy(mask[hard]);target=torch.from_numpy(gold[hard]).long()
    # Preserve the original weighted-sum reduction even though both weights=1.
    question_weights=torch.where(reference.argmax(1)!=target,1.,1.)
    members=[]
    with (directory/'head-history.csv').open('w',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['seed','epoch','train_loss']);writer.writeheader()
        for seed in settings['seeds']:
            torch.manual_seed(seed)
            network=nn.Linear(x.shape[-1],1,bias=False);nn.init.zeros_(network.weight)
            optimizer=torch.optim.AdamW(network.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
            generator=torch.Generator().manual_seed(seed)
            for epoch in range(1,settings['epochs']+1):
                losses=[]
                for batch in torch.randperm(len(x),generator=generator).split(settings['batch_size']):
                    optimizer.zero_grad(set_to_none=True)
                    delta=settings['delta_limit']*torch.tanh(network(x[batch]).squeeze(-1)/settings['delta_limit'])
                    belongs=(tid[batch,:,None]==torch.arange(4))&(tid[batch,:,None]>=0)
                    delta=(delta[:,:,None]*belongs).sum(1)/belongs.sum(1).clamp_min(1)
                    logp=F.log_softmax((reference[batch]+delta).masked_fill(~tmask[batch],-1e9),dim=1)
                    p=logp.exp()*tmask[batch]
                    nll=-logp.gather(1,target[batch,None]).squeeze(1)
                    ref_logp=F.log_softmax(reference[batch].masked_fill(~tmask[batch],-1e9),dim=1).detach()
                    kl=(p*(logp-ref_logp)*tmask[batch]).sum(1)
                    wrong=tmask[batch]&~F.one_hot(target[batch],4).bool()
                    margins=settings['ranking_margin']+logp-logp.gather(1,target[batch,None])
                    ranking=(F.softplus(margins)*wrong).sum(1)/wrong.sum(1).clamp_min(1)
                    loss=((nll+settings['kl_weight']*kl+settings['ranking_weight']*ranking)*question_weights[batch]).sum()/question_weights[batch].sum().clamp_min(1e-12)
                    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss.')
                    loss.backward();nn.utils.clip_grad_norm_(network.parameters(),settings['gradient_clip']);optimizer.step()
                    losses.append(float(loss.detach()))
                writer.writerow(dict(seed=seed,epoch=epoch,train_loss=float(np.mean(losses))));stream.flush()
                if epoch==1 or epoch%10==0:print(f'[head seed={seed} {epoch}/{settings["epochs"]}] loss={np.mean(losses):.6f}',flush=True)
            members.append(dict(linear_weight=network.weight.detach().numpy()[0].copy()))
    return shared,members


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'train_config.json')
    parser.add_argument('--output',type=Path,default=HERE/'runs/retrained_model')
    parser.add_argument('--tree-python',type=Path,help='Optional separate interpreter containing XGBoost.')
    parser.add_argument('--tree-worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--tree-output',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args();config_path=args.config.resolve();settings=read(config_path)
    if args.tree_worker:tree_worker(args.tree_worker,args.tree_output,settings['base']);return
    import torch
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    output=args.output.resolve();work=output.parent/(output.name+'-training')
    if output.exists() or work.exists():parser.error('Output or training-log directory exists; choose another --output.')
    interpreter=choose_tree_python(args.tree_python)
    started=time.perf_counter();work.mkdir(parents=True)
    plan=dict(recipe=settings,fit_split='train',validation_loaded=False,test_loaded=False,
        teacher_cache_used=False,existing_checkpoint_used=False,threads=4,tree_interpreter=str(interpreter),
        note='Frozen hyperparameters from prior development; this run performs no validation/test selection.',
        environment=dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,
                         platform=platform.platform(),device='CPU'),
        source_sha256={n:sha(HERE/n) for n in ('train.py','submission.py','train_config.json')})
    write(work/'plan.json',plan)
    bank,arrays=load_inputs('train',DATA);embeddings=arrays['embeddings']
    labels=labels_for(bank,read_jsonl(DATA/'gsm8k_train.jsonl'))
    ids,mask=answer_groups(bank);gold=ids[np.arange(len(bank)),labels.argmax(1)].astype(np.int64)
    all_indices=np.arange(len(bank))
    print('Training the full base and three OOF teachers from train inputs/references only.',flush=True)
    base,base_config=fit_base(bank,embeddings,labels,all_indices,settings['base'],work/'reference',interpreter,config_path)
    features=np.empty((len(bank),4,20),np.float32);scores=np.empty((len(bank),4),np.float32);seen=np.zeros(len(bank),np.int32)
    audits=[]
    for fold,held in enumerate(make_folds(mask.sum(1),settings['folds'],settings['fold_seed'])):
        fit=np.setdiff1d(all_indices,held)
        teacher,teacher_config=fit_base(bank,embeddings,labels,fit,settings['base'],work/f'inner{fold}',interpreter,config_path)
        f,s,group,_=prepare_inputs([bank[i] for i in held],embeddings[held],teacher,teacher_config)
        np.testing.assert_array_equal(group,ids[held]);features[held]=f;scores[held]=s;seen[held]+=1
        assert not np.intersect1d(fit,held).size
        audits.append(dict(fold=fold,fit_indices=fit.tolist(),predicted_indices=held.tolist()))
    np.testing.assert_array_equal(seen,1)
    write(work/'oof-audit.json',dict(folds=audits,each_question_scored_once=True))
    shared,members=fit_heads((features,scores,ids,mask),gold,settings['head'],work)
    weights={'base_'+key:value for key,value in base.items()};weights.update(shared);entries=[]
    for i,member in enumerate(members):
        prefix=f'head{i}_';weights.update({prefix+key:value for key,value in member.items()})
        entries.append(dict(prefix=prefix,weight=1/len(members)))
    head=settings['head']
    config=dict(format='new_disagreement_v1',protocol=PROTOCOL,fit_split='train',selection='train_nested_folds',
        input_features='evidence_arithmetic',base_config=base_config,members=entries,comparison_size=0,
        feature_clip=head['feature_clip'],delta_limit=head['delta_limit'],reference_temperature=head['reference_temperature'],
        blend_weight=head['blend_weight'],trained_epoch=head['epochs'],training_method='evidence_arithmetic',
        description='Correction with fixed epoch and uniform seeds selected on train folds')
    output.mkdir();np.savez(output/'model.npz',**weights);write(output/'config.json',config)
    assets=sum((output/n).stat().st_size for n in ('model.npz','config.json'))
    if assets>16*2**20:raise ValueError('Inference assets exceed 16 MiB.')
    report=dict(status='completed',fit_split='train',questions=len(bank),head_training_questions=int((mask.sum(1)>=2).sum()),
        teacher_models_trained=1+settings['folds'],head_seeds=head['seeds'],head_epochs=head['epochs'],
        seconds=time.perf_counter()-started,asset_bytes=assets,validation_loaded=False,test_loaded=False,
        checkpoint=str(output),sha256={n:sha(output/n) for n in ('model.npz','config.json')})
    write(work/'summary.json',report);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
