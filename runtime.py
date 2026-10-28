"""Runtime helpers: auto-detect GPU, offline-safe demo model, optional real HF model."""
from pathlib import Path
import os
import torch
OUT=Path(__file__).resolve().parent/"outputs"; OUT.mkdir(exist_ok=True)
def device_name():
 if torch.cuda.is_available(): return "cuda"
 if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return "mps"
 return "cpu"
class ByteTokenizer:
 vocab_size=256; pad_token_id=0; eos_token_id=None; eos_token=None; pad_token=None
 def __call__(self,text,return_tensors="pt",**kwargs):
  ids=[ord(c)%256 for c in text[:120]] or [1]
  return {"input_ids":torch.tensor([ids],dtype=torch.long)} if return_tensors=="pt" else {"input_ids":ids}
 def decode(self,ids,skip_special_tokens=True):
  if isinstance(ids,torch.Tensor): ids=ids.tolist()
  return "".join(chr(int(i)%256) for i in ids if int(i)!=0)
class Bundle:
 def __init__(self,model,tokenizer,model_id,device): self.model=model; self.tokenizer=tokenizer; self.model_id=model_id; self.device=device
def load_bundle(real_model=False,model_id=None):
 from transformers import GPT2Config,GPT2LMHeadModel,AutoTokenizer,AutoModelForCausalLM
 device=device_name(); chosen=model_id or os.environ.get("NEURONLAB_MODEL",("distilgpt2" if real_model else "demo-random-gpt2"))
 if not real_model and chosen=="demo-random-gpt2":
  tok=ByteTokenizer(); cfg=GPT2Config(vocab_size=256,n_positions=128,n_ctx=128,n_embd=64,n_layer=2,n_head=2)
  torch.manual_seed(7); model=GPT2LMHeadModel(cfg); chosen="demo-random-gpt2"
 else:
  tok=AutoTokenizer.from_pretrained(chosen,use_fast=True)
  model=AutoModelForCausalLM.from_pretrained(chosen,torch_dtype=torch.float16 if device=="cuda" else torch.float32,low_cpu_mem_usage=True)
  if tok.pad_token_id is None and tok.eos_token_id is not None: tok.pad_token=tok.eos_token
 model.to(device).eval(); return Bundle(model,tok,chosen,device)
def forward_text(bundle,text):
 batch=bundle.tokenizer(text,return_tensors="pt"); batch={k:v.to(bundle.device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
 with torch.no_grad(): out=bundle.model(**batch,output_hidden_states=True,output_attentions=True,return_dict=True)
 return batch,out
def generate(bundle,prompt,max_new_tokens=48):
 batch=bundle.tokenizer(prompt,return_tensors="pt"); batch={k:v.to(bundle.device) for k,v in batch.items() if isinstance(v,torch.Tensor)}
 with torch.no_grad(): ids=bundle.model.generate(**batch,max_new_tokens=max_new_tokens,do_sample=False,pad_token_id=getattr(bundle.tokenizer,"pad_token_id",None))
 return bundle.tokenizer.decode(ids[0][batch["input_ids"].shape[-1]:],skip_special_tokens=True)
