const SIZE=96, SR=96;
export const policy=Object.freeze({size:SIZE,sampleRate:SR,trainBatches:18,testBatches:10,
  windowsPerBatch:24,modelCutoff:.5,seed:3221});
const conditions=['normal','isolated','chatter','load'];
export const names={normal:'穩定加工',isolated:'換料／短時擾動',chatter:'高頻顫振',load:'機械負載偏移',unsupported:'取樣條件不符'};
const rnd=seed=>()=>{seed=(Math.imul(seed,1664525)+1013904223)>>>0;return seed/4294967296;};
const avg=x=>x.reduce((s,i)=>s+i,0)/x.length;
const rms=x=>Math.sqrt(avg(x.map(v=>v*v)));
export function generate(part,id,scenario=null,seed=policy.seed){
 const [set,idx]=part==='test'?[991801,policy.testBatches]:[304711,policy.trainBatches];
 const batch=Math.floor(id/policy.windowsPerBatch);
 if(!Number.isSafeInteger(id)||id<0||batch>=idx||!['test','train'].includes(part))throw Error('樣本範圍錯誤');
 const shared=rnd((seed+set+Math.imul(batch+1,375617))>>>0);const material=shared();
 const batchScale=.75+.44*material+(part==='test'?.09:0), rpm=1690+240*shared(), phase=shared()*6.28;
 const rand=rnd((seed+set+Math.imul(id+1,677891))>>>0);
 const order=(id%24+batch*7)%24;
 const kind=scenario ?? (order<10?'normal':order<15?'isolated':order<20?'chatter':'load');
 if(![...conditions,'unsupported'].includes(kind))throw Error('未知情境');
 const v=[],amps=[],speeds=[];
 const amplitude=.33+.46*rand(), drive=3.1+1.15*rand(), strength=.3+.99*rand(), blur=.8+rand()*.75;
 const eventAt=.15+.50*rand(), winding=25+Math.floor(rand()*8), imp=rand()<.5?-1:1;
 for(let i=0;i<SIZE;i++){
  let t=i/SR;
  let sample=amplitude*batchScale*Math.sin(2*Math.PI*(5+Math.round(material*3))*t+phase)+(.39*rand()-.195)*blur;
  let current=drive+.14*Math.sin(t*9+phase)+(.22*rand()-.11)*blur;
  let speed=rpm+12*Math.sin(t*7+phase)+(rand()-.5)*14;
  if(kind==='isolated') {
    const transient=Math.exp(-((t-eventAt)**2)/.0011);
    sample+=imp*(1.9+strength)*transient; current+=.7*transient; speed-=26*transient;
  }
  if(kind==='chatter') {
    const onset=.6+.4*Math.sin(t*3.14);
    sample+=strength*.52*onset*Math.sin(t*2*Math.PI*winding+phase);
    current+=.22*strength*t; speed-=21*strength*t;
  }
  if(kind==='load') {
    const onset=Math.max(0,t-eventAt);
    const rise=(1.4+1.2*strength)*onset, dip=(100+102*strength)*onset;
    current+=rise; speed-=dip; sample+=.23*strength*Math.sin(t*18+phase);
  }
  v.push(sample);amps.push(current);speeds.push(speed);
 }
 return {sampleId:part+'-'+id,batchId:part+'-'+String(batch+1).padStart(2,'0'),kind,
  label:kind==='unsupported'?null:kind==='chatter'||kind==='load'?1:0,
  sampleRate:kind==='unsupported'?48:SR,vibration:v,current:amps,rpm:speeds};
}
export function rejectReason(w){
 if(w?.sampleRate!==SR||!['vibration','current','rpm'].every(k=>Array.isArray(w[k])&&w[k].length===SIZE))return '取樣頻率或點數不符';
 const bounds={vibration:[-4,4,.08],current:[1,8,.05],rpm:[1300,2200,5]};
 for(const [k,[low,high,minimumSpan]] of Object.entries(bounds)){
  const a=w[k];
  if(a.some(v=>!Number.isFinite(v)||v<low||v>high))return '數值缺失或超出合成操作範圍';
  if(Math.max(...a)-Math.min(...a)<minimumSpan)return '訊號平線或量測中斷';
 }
 return null;
}
export function features(w){
 if(rejectReason(w))return null;
 const v=w.vibration, halves=a=>avg(a.slice(-24))-avg(a.slice(0,24));
 const delta=v.slice(1).map((x,i)=>x-v[i]);
 return [rms(v), rms(delta)/(rms(v)+.001), Math.abs(halves(w.current)), Math.abs(halves(w.rpm))];
}
function all(part,n){return Array.from({length:n*policy.windowsPerBatch},(_,i)=>generate(part,i));}
export function train(windows){
 const examples=windows.map(w=>({x:features(w),y:w.label}));
 if(!examples.length||examples.some(e=>!e.x||![0,1].includes(e.y))||!examples.some(x=>x.y===0)||!examples.some(x=>x.y===1))throw Error('訓練資料條件不足');
 const means=examples[0].x.map((_,i)=>avg(examples.map(e=>e.x[i])));
 const scale=means.map((m,i)=>Math.sqrt(avg(examples.map(e=>(e.x[i]-m)**2)))||1);
 const data=examples.map(e=>({x:[1,...e.x.map((v,i)=>(v-means[i])/scale[i])],y:e.y}));
 const weights=[0,0,0,0,0],sig=x=>1/(1+Math.exp(-x));
 for(let epoch=0;epoch<550;epoch++){
  const g=[0,0,0,0,0];
  for(const e of data){
   const error=sig(e.x.reduce((s,x,i)=>s+x*weights[i],0))-e.y;
   for(let j=0;j<g.length;j++)g[j]+=error*e.x[j];
  }
  for(let j=0;j<g.length;j++)weights[j]-=.12*(g[j]/data.length+(j?.035*weights[j]:0));
 }
 const candidates=[...new Set(examples.map(e=>e.x[0]))].sort((a,b)=>a-b);
 const pos=examples.filter(e=>e.y===1).length,neg=examples.length-pos;
 let best=-1,baseline=candidates[0];
 for(const cutoff of candidates){
  let tp=0,tn=0;
  for(const e of examples){if(e.y&&e.x[0]>=cutoff)tp++;if(!e.y&&e.x[0]<cutoff)tn++;}
  const bal=(tp/pos+tn/neg)/2;
  if(bal>best){best=bal;baseline=cutoff;}
 }
 // Strong comparator: train-only 2D OR rule, grid of every fourth sorted
 // training value, plus a disabled branch above the maximum. No holdout tuning.
 const grid=i=>{
  const a=examples.map(e=>e.x[i]).sort((a,b)=>a-b);
  return [...new Set(a.filter((_,k)=>k%4===0)),a[a.length-1]+1];
 };
 let ruleBest=-1,rule=null;
 for(const a of grid(1))for(const b of grid(3)){
  let tp=0,tn=0;
  for(const e of examples){
   const flagged=e.x[1]>=a||e.x[3]>=b;
   if(e.y&&flagged)tp++;if(!e.y&&!flagged)tn++;
  }
  const bal=(tp/pos+tn/neg)/2;
  if(bal>ruleBest){ruleBest=bal;rule={quickRatio:a,rpmDelta:b};}
 }
 return {means,scale,weights,baseline,rule};
}
export function classify(model,window){
 const x=features(window);
 if(!x||!x.every(Number.isFinite))return {verdict:'unsupported',score:null,baseline:null,rule:null,features:null,reason:rejectReason(window)||'特徵計算失敗'};
 const z=x.map((v,i)=>(v-model.means[i])/model.scale[i]);
 const score=1/(1+Math.exp(-z.reduce((s,v,i)=>s+v*model.weights[i+1],model.weights[0])));
 return {verdict:score>=policy.modelCutoff?'review':'normal',score,baseline:x[0]>=model.baseline?'review':'normal',
  rule:x[1]>=model.rule.quickRatio||x[3]>=model.rule.rpmDelta?'review':'normal',features:x,reason:null};
}
export function evaluate(windows,model){
 const stats=()=>({tp:0,tn:0,fp:0,fn:0,unsupported:0});
 const logistic=stats(),baseline=stats(),rule=stats(),breakdown={};
 const add=(y,p,c)=>{if(!p||y===null||p==='unsupported'){c.unsupported++;return;}
   const flag=p==='review';c[y?(flag?'tp':'fn'):(flag?'fp':'tn')]++;};
 for(const w of windows){
  const result=classify(model,w);
  if(!breakdown[w.kind])breakdown[w.kind]=stats();
  add(w.label,result.verdict,logistic);add(w.label,result.baseline,baseline);add(w.label,result.rule,rule);add(w.label,result.verdict,breakdown[w.kind]);
 }
 const summarize=c=>{
   const ratio=(n,d)=>d?n/d:null;
   const recall=ratio(c.tp,c.tp+c.fn), falseAlarm=ratio(c.fp,c.fp+c.tn);
   return {...c,recall,falseAlarm,precision:ratio(c.tp,c.tp+c.fp),
    balanced:recall===null||falseAlarm===null?null:(recall+1-falseAlarm)/2};
 };
 return {logistic:summarize(logistic),baseline:summarize(baseline),rule:summarize(rule),
  breakdown:Object.fromEntries(Object.entries(breakdown).map(([k,v])=>[k,summarize(v)]))};
}
export function run(){
 const trainRows=all('train',policy.trainBatches),holdout=all('test',policy.testBatches),model=train(trainRows);
 const metrics=evaluate(holdout,model);
 return {model,metrics,batches:{train:[...new Set(trainRows.map(w=>w.batchId))],test:[...new Set(holdout.map(w=>w.batchId))]},windows:{train:trainRows.length,test:holdout.length}};
}
