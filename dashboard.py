"""Streamlit dashboard for Experiment Scientist. Run: streamlit run dashboard.py"""
import streamlit as st
import pandas as pd
from prompt_experiment import run
st.set_page_config(page_title="Experiment Scientist",layout="wide")
st.title("Experiment Scientist")
st.caption("Paired prompt experiments. Built-in examples are demonstrations, not pre-registered scientific evidence.")
model=st.text_input("Model ID","distilgpt2")
device=st.selectbox("Device",["auto","cuda","mps","cpu"])
tokens=st.slider("Maximum new tokens",32,256,120,step=16)
seed=st.number_input("Seed",min_value=0,max_value=100000,value=17)
if st.button("Run paired experiment",type="primary"):
    with st.spinner("Running baseline and treatment arms…"):
        summary=run(model,device=device,max_new_tokens=tokens,seed=int(seed))
    st.metric("Prompt pairs",summary["pairs"])
    st.metric("Mean word-count delta",f"{summary['mean_word_delta']:+.2f}")
    payload=__import__("json").loads(open("outputs/experiment/experiment.json",encoding="utf-8").read())
    ci=payload.get("bootstrap_word_delta", {})
    if ci.get("lower") is not None:
        st.caption(f"Exploratory 95% bootstrap interval for mean word-count delta: [{ci['lower']:.2f}, {ci['upper']:.2f}] (small sample; interpret cautiously).")
    rows=[]
    for r in payload["results"]:
        rows.append({"id":r["id"],"baseline_words":r["baseline"]["words"],"treatment_words":r["treatment"]["words"],"delta":r["word_delta_treatment_minus_baseline"]})
    st.dataframe(pd.DataFrame(rows),use_container_width=True)
    with st.expander("Inspect paired outputs"):
        for r in payload["results"]:
            st.subheader(r["id"]); st.markdown("**Baseline**");st.write(r["baseline"]["response"]);st.markdown("**Treatment**");st.write(r["treatment"]["response"])
