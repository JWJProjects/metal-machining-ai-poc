import {policy,names,run,generate,classify} from './model.mjs';
const $=x=>document.getElementById(x);
const report=run();
const features=['振動 RMS','快速變化比','電流前後差（A）','轉速前後差（rpm）'];
const channels={vibration:'主軸振動（任意單位）',current:'驅動電流（模擬 A）',rpm:'主軸轉速（模擬 rpm）'};
const scales={vibration:[-4,4],current:[1,8],rpm:[1300,2200]};
let active;
const opt=(value,text)=>Object.assign(document.createElement('option'),{value,textContent:text});
for(const k of ['chatter','load','isolated','normal','unsupported']) $('scenario').append(opt(k,names[k]));
for(let i=0;i<policy.windowsPerBatch*policy.testBatches;i++) {
 const w=generate('test',i);
 if(w.label===1&&classify(report.model,w).verdict==='normal'){
   $('scenario').append(opt('holdout-'+i,'留出測試漏報案例（'+w.sampleId+'）'));break;
 }
}
const text=(id,value)=>$(id).textContent=value;
const fmt=x=>x===null?'不適用':(x*100).toFixed(1)+'%';
const resultText=x=>({review:'建議人工覆核',normal:'不列優先覆核',unsupported:'資料不支援'})[x]||'不評分';
function plot(w,key,supported){
 text('plot-title',channels[key]);text('sample-id',w.batchId+(supported?' · '+w.sampleRate+' Hz':' · 不支援'));
 const root=$('chart');root.textContent='';
 if(!supported){text('chart-caption','取樣條件不符，略過趨勢與評分。原始樣本仍可在匯出中查閱。');return;}
 const values=w[key], [low,high]=scales[key], ns='http://www.w3.org/2000/svg';
 const x=(i)=>61+i*(535/(values.length-1)),y=(v)=>195-(v-low)*150/(high-low);
 const mark=(tag,attrs,value='')=>{const el=document.createElementNS(ns,tag);for(const [k,v] of Object.entries(attrs))el.setAttribute(k,v);if(value)el.textContent=value;root.append(el);return el;};
 for(let i=0;i<5;i++){
  const v=low+i*(high-low)/4, pos=y(v);
  mark('line',{x1:61,y1:pos,x2:596,y2:pos,class:'grid'});
  mark('text',{x:54,y:pos+3,'text-anchor':'end',class:'axis'},v.toFixed(key==='rpm'?0:1));
  mark('line',{x1:61+535*i/4,y1:45,x2:61+535*i/4,y2:195,class:'grid'});
  mark('text',{x:61+535*i/4,y:221,'text-anchor':'middle',class:'axis'},i===4?'約 1 秒':(i/4).toFixed(2));
 }
 mark('polyline',{points:values.map((v,i)=>x(i).toFixed(2)+','+y(v).toFixed(2)).join(' '),class:'wave'});
 text('chart-caption','同一時窗、固定訊號軸。縱軸為各訊號模擬單位，橫軸為秒。');
}
function update(reset=true){
 const scenario=$('scenario').value;
 const holdout=scenario.startsWith('holdout-');
 const w=holdout?generate('test',Number(scenario.slice(8))):{...generate('test',32,scenario),sampleId:'demo-'+scenario,batchId:'合成展示'};
 const result=classify(report.model,w);
 active={w,result,source:holdout?'held_out':'interactive_synthetic'};
 if(reset)$('review').value='';
 plot(w,$('channel').value,result.verdict!=='unsupported');
 text('decision-title',resultText(result.verdict));$('decision-title').dataset.state=result.verdict;
 text('confidence',result.score===null?'不產生分數：'+result.reason+'。':'模型分流分數 '+result.score.toFixed(3)+' ／ 門檻 '+policy.modelCutoff.toFixed(2)+'。分數不是故障機率。');
 text('truth',names[w.kind]);
 text('baseline',resultText(result.baseline));
 text('rules',resultText(result.rule));
 text('feature-values',result.features?result.features.map((x,i)=>features[i]+' '+x.toFixed(2)).join('、'):'不適用');
 text('announcement',names[w.kind]+'：'+resultText(result.verdict));
}
$('scenario').addEventListener('change',()=>update());
$('channel').addEventListener('change',()=>update(false));
const row=(label,r)=>{
 const tr=document.createElement('tr'), name=document.createElement('th');
 name.scope='row';name.textContent=label;tr.append(name);
 for(const value of [fmt(r.recall),fmt(r.falseAlarm),r.tp+' / '+r.fn,r.tn+' / '+r.fp]){const td=document.createElement('td');td.textContent=value;tr.append(td);}
 $('results').append(tr);
};
const m=report.metrics;
text('cohort',report.windows.train+' 窗／'+report.batches.train.length+' 批次訓練 · '+report.windows.test+' 窗／'+report.batches.test.length+' 個另批留出，以批次為界。');
row('多訊號模型',m.logistic);row('雙訊號 OR 規則',m.rule);row('振動 RMS 基線',m.baseline);
text('limits','多訊號模型漏報 '+m.logistic.fn+' 窗，其中顫振 '+m.breakdown.chatter.fn+' 窗。誤報 '+m.logistic.fp+'/'+(m.logistic.fp+m.logistic.tn)+' 只代表本次合成樣本。雙訊號規則較少漏報但多 1 筆誤報，本比較不能外推至其他設備、材料或真實異常。');
$('export').addEventListener('click',()=>{
 const content={schema_version:1,provenance:'synthetic_offline',policy,report,active:{...active,manual_review:$('review').value||null},
 limitations:['sample_accuracy_not_real_world','not_machine_health_assessment','not_machine_control_or_safety','single_simulation_family']};
 const url=URL.createObjectURL(new Blob([JSON.stringify(content,null,2)+'\n'],{type:'application/json'}));
 const a=document.createElement('a');a.href=url;a.download='轉序-合成測試紀錄.json';document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
 text('announcement','已產生合成驗證紀錄。');
});
update();
