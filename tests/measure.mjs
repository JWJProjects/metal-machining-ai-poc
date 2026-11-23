import {run,generate,evaluate,policy} from '../model.mjs';
const r=run();
const baselineWindows=Array.from({length:r.windows.test},(_,i)=>generate('test',i));
const evaluateStress=(mutate)=>evaluate(baselineWindows.map(mutate),r.model);
const draft={
 schema_version:1,
 provenance:'synthetic_offline',
 policy,
 main:r,
 stress_existing_holdout_only:{
  vibration_gain_045:evaluateStress(s=>({...s,vibration:s.vibration.map(x=>x*.45)})),
  vibration_gain_180:evaluateStress(s=>({...s,vibration:s.vibration.map(x=>x*1.8)})),
  rpm_linear_drift_22:evaluateStress(s=>({...s,rpm:s.rpm.map((x,i)=>x+22*i/(s.rpm.length-1))})),
 },
 limitations:['all_records_generated_from_one_simulation_family','stress_not_a_second_independent_dataset','thresholds_not_field_specifications','not_health_assessment_control_or_safety'],
};
console.log(JSON.stringify(draft,null,2));
