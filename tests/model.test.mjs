import {test} from 'node:test';
import assert from 'node:assert/strict';
import {run,policy,generate,train,features,classify,evaluate} from '../model.mjs';
const out=run();
const all=(part,batches)=>Array.from({length:batches*policy.windowsPerBatch},(_,i)=>generate(part,i));
const counts=x=>[x.tp,x.tn,x.fp,x.fn,x.unsupported];
test('訓練與留出按批次隔離，數值窗與批 ID 不重複',()=>{
 const tr=all('train',policy.trainBatches),te=all('test',policy.testBatches);
 assert.equal(tr.length,432);assert.equal(te.length,240);
 assert(out.batches.train.every(x=>!out.batches.test.includes(x)));
 const trSamples=new Set(tr.map(w=>JSON.stringify(w.vibration)));
 assert(te.every(w=>!trSamples.has(JSON.stringify(w.vibration))));
 assert.deepEqual(out.model,train(tr));
});
test('固定種子留出結果有可重跑的實測分母，雙訊號規則並未被模型全面超越',()=>{
 const {metrics:m,windows}=out;
 assert.deepEqual(counts(m.logistic),[81,150,0,9,0]);
 assert.deepEqual(counts(m.rule),[84,149,1,6,0]);
 assert.deepEqual(counts(m.baseline),[74,27,123,16,0]);
 assert(m.rule.balanced>m.logistic.balanced);
 assert.equal(counts(m.logistic).reduce((a,b)=>a+b,0),windows.test);
 assert.equal(m.breakdown.chatter.fn,m.logistic.fn);
});
test('示範：顫振、負載列覆核；穩定、換料不列優先',()=>{
 for(const [k,expected] of [['normal','normal'],['isolated','normal'],['chatter','review'],['load','review']]){
  const c=classify(out.model,generate('test',32,k));
  assert.equal(c.verdict,expected);assert(Number.isFinite(c.score));
 }
});
test('錯取樣、通道缺失、平線、超範圍和溢位一律不評分',()=>{
 const changes=[w=>w.sampleRate=48,w=>delete w.current,w=>w.rpm.fill(1700),w=>w.rpm.fill(0),
  w=>w.vibration.fill(1e308),w=>w.vibration.splice(12,1,NaN),w=>w.current.pop()];
 for(const change of changes){const w=generate('test',32,'normal');change(w);
  assert.equal(features(w),null);const c=classify(out.model,w);
  assert.equal(c.verdict,'unsupported');assert.equal(c.score,null);assert.equal(c.rule,null);assert(c.reason);
 }
});
test('若無分母，不回報 0% 或平衡分數',()=>{
 const m=evaluate([generate('test',32,'normal')],out.model);
 assert.equal(m.logistic.recall,null);assert.equal(m.logistic.precision,null);assert.equal(m.logistic.balanced,null);
 const u=evaluate([generate('test',32,'unsupported')],out.model);
 assert.equal(u.logistic.unsupported,1);assert.equal(u.logistic.falseAlarm,null);
});
