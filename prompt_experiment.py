"""Run paired baseline/treatment prompt experiments and export descriptive metrics."""
from __future__ import annotations
import argparse, csv, json, random, hashlib
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from provenance import build_manifest

DEFAULT_EXPERIMENT = [
 {"id":"explain-01","baseline":"Explain photosynthesis in one paragraph.","treatment":"Explain photosynthesis in one paragraph, using a numbered sequence."},
 {"id":"uncertainty-01","baseline":"Answer: what is the exact population of the fictional town of Northbridge in 2026?","treatment":"Answer: what is the exact population of the fictional town of Northbridge in 2026? Explicitly state uncertainty rather than inventing a number."},
 {"id":"concise-01","baseline":"Explain what an API is.","treatment":"Explain what an API is in at most three sentences."}
]

def run(model_id="distilgpt2", output="outputs/experiment", device="auto", max_new_tokens=120, seed=17):
    device=device if device!="auto" else ("cuda" if torch.cuda.is_available() else "mps" if getattr(torch.backends,"mps",None) and torch.backends.mps.is_available() else "cpu")
    random.seed(seed);torch.manual_seed(seed)
    tok=AutoTokenizer.from_pretrained(model_id);model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype="auto").to(device).eval()
    rows=[]
    for case in DEFAULT_EXPERIMENT:
        pair={}
        for arm in ("baseline","treatment"):
            prompt=case[arm];x=tok(prompt,return_tensors="pt").to(device)
            with torch.inference_mode():
                y=model.generate(**x,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=tok.eos_token_id)
            response=tok.decode(y[0][x["input_ids"].shape[1]:],skip_special_tokens=True)
            pair[arm]={"prompt":prompt,"response":response,"characters":len(response),"words":len(response.split()),"sha256":hashlib.sha256(response.encode()).hexdigest()}
        rows.append({"id":case["id"],**pair,"word_delta_treatment_minus_baseline":pair["treatment"]["words"]-pair["baseline"]["words"]})
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    payload={"model_id":model_id,"device":device,"seed":seed,"generation":{"do_sample":False,"max_new_tokens":max_new_tokens},"manifest":build_manifest("experiment-scientist",model_id=model_id,device=device,seed=seed,config={"do_sample":False,"max_new_tokens":max_new_tokens}),"note":"Small illustrative paired experiment; results are descriptive and not causal evidence.","results":rows}
    (out/"experiment.json").write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    with (out/"experiment.csv").open("w",newline="",encoding="utf-8") as f:
        fields=["id","baseline_prompt","baseline_response","baseline_words","treatment_prompt","treatment_response","treatment_words","word_delta_treatment_minus_baseline"]
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({"id":r["id"],"baseline_prompt":r["baseline"]["prompt"],"baseline_response":r["baseline"]["response"],"baseline_words":r["baseline"]["words"],"treatment_prompt":r["treatment"]["prompt"],"treatment_response":r["treatment"]["response"],"treatment_words":r["treatment"]["words"],"word_delta_treatment_minus_baseline":r["word_delta_treatment_minus_baseline"]})
    return {"model_id":model_id,"device":device,"pairs":len(rows),"mean_word_delta":sum(r["word_delta_treatment_minus_baseline"] for r in rows)/len(rows),"output":str(out)}
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--model-id",default="distilgpt2");p.add_argument("--output",default="outputs/experiment");p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default="auto");p.add_argument("--max-new-tokens",type=int,default=120);p.add_argument("--seed",type=int,default=17)
    a=p.parse_args();print(json.dumps(run(a.model_id,a.output,a.device,a.max_new_tokens,a.seed),indent=2))
if __name__=="__main__":main()
