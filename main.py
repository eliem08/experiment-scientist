import argparse, json
from app import run
p=argparse.ArgumentParser(description="Experiment Scientist")
p.add_argument("--text",default=None); p.add_argument("--real-model",action="store_true"); p.add_argument("--model-id",default=None)
a=p.parse_args(); result=run(text=a.text,real_model=a.real_model,model_id=a.model_id)
print(json.dumps(result,indent=2,ensure_ascii=False,default=str))
