import json
from datetime import datetime,timezone
from runtime import load_bundle, generate, OUT

def run(text=None,real_model=False,model_id=None):
 b=load_bundle(real_model=real_model,model_id=model_id)
 question=text or "Does an explicit two-sentence limit reduce output length compared with a direct baseline?"
 conditions=[
  ("baseline",question),
  ("treatment","Answer in at most two sentences. "+question),
  ("control","Before answering, consider this unrelated phrase: cobalt ladder tulip. "+question),
 ]
 results=[]
 for name,prompt in conditions:
  output=generate(b,prompt,max_new_tokens=80)
  results.append({"condition":name,"prompt":prompt,"output":output,"characters":len(output),"words":len(output.split())})
 report={"timestamp":datetime.now(timezone.utc).isoformat(),"model":b.model_id,"device":b.device,"research_question":question,"conditions":results,"interpretation":"Exploratory run only. Predefine outcomes and repeat on held-out prompts before making claims."}
 p=OUT/"experiment_report.json"; p.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8"); report["path"]=str(p); return report
