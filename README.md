# 🧠 MARAG: ReAct-based Multimodal RAG Pipeline

This repository implements a fully agentic, multimodal RAG system. The pipeline relies on a ReAct loop orchestrating Qwen2.5-VL-3B-Instruct, combined with Multimodal Query Fusion (EVA-CLIP) and Section-Level Relevance Filtering (ReAG-Critic).

## 🏗️ System Architecture Diagram

```text
┌─────────────────────────────────────────────────────────────────────────┐
│                        📸 INPUT: Image + Question                       │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ 🤖 SMOLAGENTS AGENT LOOP (qwen_agent_model.py + run_inference_agent.py) │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ 🛠️ TOOL 1: assess_retrieval_need (agent_tools.py)                 │  │
│  │  • Dedicated greedy Qwen call evaluating only Image + Question    │  │
│  │  • Forces agent to decide between [RETRIEVE] or [ANSWER_DIRECTLY] │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
│                                    ▼ (If RETRIEVE is chosen)            │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ 🛠️ TOOL 2: retrieve_knowledge (agent_tools.py & retriever_agent)  │  │
│  │                                                                   │  │
│  │  (A) Guesses Generation                                           │  │
│  │      Dedicated Qwen call -> Predicts Top-3 entity hypotheses      │  │
│  │             │                                                     │  │
│  │  (B) Multimodal Query Fusion                                      │  │
│  │      Query = "Image tags: [Guesses]. Question: [Q]"               │  │
│  │      EVA-CLIP-8B embeds -> 70% Image Features + 30% Text Features │  │
│  │             │                                                     │  │
│  │  (C) FAISS Search                                                 │  │
│  │      Retrieves Top-8 FULL Wikipedia Documents                     │  │
│  │             │                                                     │  │
│  │  (D) Section-Level Filtering (reag_critic.py)                     │  │
│  │      ReAG-Critic scores EVERY single section individually.        │  │
│  │      Only sections scoring above threshold (Yes) are kept.        │  │
│  │                                                                   │  │
│  │  • Output: Formatted, relevant sections returned to Agent Memory  │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
│                                    ▼                                    │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ 🏁 ACTION: final_answer                                           │  │
│  │  • Agent reads the filtered context & generates a short answer    │  │
│  └─────────────────────────────────┬─────────────────────────────────┘  │
└────────────────────────────────────┼────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼────────────────────────────────────┐
│ 🛡️ CODE-LEVEL FALLBACK INTERCEPTOR (run_inference_agent.py)           │
│  • Triggered if the Agent answers with bail-outs (e.g. "Unknown")       │
│  • Triggered if ReAG-Critic filtered out ALL retrieved sections         │
│  => Overrides prediction with Plain-VLM Fallback (Image + Question)     │
└────────────────────────────────────┬────────────────────────────────────┘
                                     ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           ✨ FINAL PREDICTION                           │
└─────────────────────────────────────────────────────────────────────────┘
