# app.py

# Component 1 — Load the FAISS index + embedding model
import os
import warnings
import torch
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

print("=== Loading FAISS index ===")

# CRITICAL: you must use the EXACT same embedding model as ingest.py.
# Why? Because the vectors in your FAISS index were created by this model.
# If you load a different model here, the query vector lives in a different
# mathematical space — similarity search becomes meaningless gibberish.
# Think of it like using a French dictionary to look up a word you wrote in Chinese.

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # must match ingest.py
    model_kwargs={"device": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = FAISS.load_local(
    "./faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True,
)

print(f"Index loaded. Total vectors: {vectorstore.index.ntotal}")
# ntotal tells you exactly how many chunks are stored — should match ingest.py output

# Component 2 — Load the LLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline

print("\n=== Loading LLM ===")

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"

# --- What is 4-bit quantization? ---
# A full float32 model stores each parameter as 4 bytes.
# 3.8B parameters × 4 bytes = ~15 GB — too large for many GPUs.
# Quantization reduces each parameter to 4 bits (0.5 bytes).
# 3.8B parameters × 0.5 bytes = ~1.9 GB — fits easily.
# The quality loss is minimal for inference tasks like RAG.

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,

    bnb_4bit_compute_dtype=torch.float16,
    # Even though weights are stored in 4-bit,
    # the actual matrix multiplications happen in float16.
    # This gives a good speed/quality balance on your RTX 8000.

    bnb_4bit_use_double_quant=True,
    # Quantizes the quantization constants themselves — saves ~0.4 GB extra.
    # Negligible quality impact, worth enabling.

    bnb_4bit_quant_type="nf4",
    # NF4 = NormalFloat4. A 4-bit data type designed specifically for
    # neural network weights (which follow a normal distribution).
    # More accurate than standard int4 quantization.
)

print("  Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("  Loading model in 4-bit... (first time downloads ~2 GB)")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="cuda:0",  # load onto first RTX 8000
    trust_remote_code=True,
)

# Check how much VRAM it's actually using
vram_used = torch.cuda.memory_allocated(0) / 1e9
print(f"  Model loaded. VRAM used: {vram_used:.1f} GB")

# --- What is a pipeline? ---
# A HuggingFace pipeline bundles tokenizer + model + decoding logic
# into a single callable. It handles:
#   text → token IDs → model forward pass → token IDs → text
# so you just pass a string and get a string back.

llm_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,

    max_new_tokens=512,
    # Maximum number of tokens the LLM can generate in its answer.
    # 512 ≈ 350-400 words. Enough for a detailed research answer.

    do_sample=False,
    # do_sample=False means greedy decoding — always pick the most
    # probable next token. Deterministic and factual.
    # do_sample=True with temperature would give more creative answers,
    # but for a research Q&A you want consistency.

    repetition_penalty=1.1,
    # Slightly penalize repeating the same phrases.
    # LLMs sometimes loop ("The answer is... The answer is...").
    # 1.1 is a gentle nudge to avoid this.

    return_full_text=False,
    # Only return the generated answer, not the prompt echoed back.
    # Without this, the output includes your entire system prompt + question.
)

# Wrap in LangChain's interface so it plugs into the chain
llm = HuggingFacePipeline(pipeline=llm_pipeline)
print("  LLM ready.")

# Component 3 — Strict Extraction Prompt (prevents hallucinations)
from langchain.prompts import PromptTemplate

print("\n=== Setting up prompt template ===")

prompt_template = """<|system|>
You are a strict information extractor. You must ONLY reproduce information VERBATIM from the CONTEXT.

CRITICAL RULES:
1. DO NOT add any numbers, statistics, acronym that are not EXPLICITLY written in the CONTEXT
2. DO NOT guess or complete partial information
3. If information is not directly stated, say "Not explicitly stated in the provided excerpts"
4. KEEP acronyms and use them directly and DO NOT guess

EXTRACTION FORMAT:
- Write in clear, natural, human language


CONTEXT:
{context}

QUESTION:
{question}

<|assistant|>
Based on the provided excerpts:"""

prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"],
)

print("  Prompt template configured.")
print(f"  Template length: {len(prompt_template)} chars")

# Component 4 — Assemble the LangChain RAG chain
from langchain.chains import RetrievalQA

print("\n=== Assembling RAG chain ===")

# --- What is a chain? ---
# LangChain's RetrievalQA chains together:
#   1. A retriever (queries FAISS, returns chunks)
#   2. A prompt template (formats chunks + question)
#   3. An LLM (generates the answer)
# into a single object you can call with one line.



# Component 5 — Query function + transparency layer
print("\n=== Defining query function ===")

def ask(question: str, paper_name: str, verbose: bool = True) -> dict:
    # The retriever is the bridge between FAISS and the chain.
    retriever = vectorstore.as_retriever(search_type="similarity", 
                                        search_kwargs={"k": 2, "fetch_k": 10,"filter": {"filename": paper_name}})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        # "stuff" means: take all retrieved chunks and "stuff" them
        # into a single prompt. Simple and effective for small k.
        # Alternative chain types:
        # "map_reduce" — summarize each chunk separately, then combine
        # "refine"     — iteratively refine the answer chunk by chunk
        # For k=3 chunks, "stuff" is always the right choice.

        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
        # This is crucial for transparency — the chain also returns
        # WHICH chunks it retrieved, so you can verify the answer
        # actually came from your papers and not from hallucination.
    )


    """
    Ask a question about your research papers.

    Args:
        question: natural language question
        verbose: if True, prints retrieved sources before the answer

    Returns:
        dict with 'answer' and 'sources' keys
    """

    if verbose:
        print(f"\n{'='*60}")
        print(f"Question: {question}")
        print('='*60)

    # Run the full chain:
    # question → embed → FAISS search → prompt → LLM → answer
    result = qa_chain.invoke({"query": question})

    answer = result["result"]
    source_docs = result["source_documents"]

    if verbose:
        # Show which chunks were retrieved BEFORE showing the answer.
        # This is the transparency layer — you can verify the LLM
        # had access to the right context.
        print("\n--- Retrieved context ---")
        for i, doc in enumerate(source_docs):
            source = doc.metadata.get("filename") or os.path.basename(doc.metadata.get("source", "unknown"))
            section = doc.metadata.get("section", "—")
            display = doc.metadata.get("display", source)
            preview = doc.page_content[:200].replace("\n", " ")
            print(f"  Chunk {i+1}: {display}")
            print(f"  Section: {section}")
            print(f"  Text: {preview}...")

        print("--- Answer ---")
        print(answer)

        print("\n--- Sources used ---")
        seen = set()
        for doc in source_docs:
            display = doc.metadata.get("display", "unknown")
            if display not in seen:
                print(f"  {display}")
                seen.add(display)

    return {
        "answer": answer,
        "sources": source_docs,
    }


# # Component 6 — Test suite before launching the UI
# print("\n=== Running test queries ===")

# # These questions cover all 4 of your papers.
# # Run these manually first to verify the chain works
# # before exposing it via Gradio.

# test_questions = [
#     # From Neural Networks 2026 paper
#     "What is rank-order coding and how does it relate to neural sequences?",

#     # From IEEE ASRU 2025 paper
#     "What dataset was used for the EMG-to-speech experiments?",

#     # From ICANN 2024 paper
#     "How does the predictive coding model handle bilingual vocal learning?",

#     # From ICDL 2022 paper
#     "How were gain-field networks used for visuo-motor remapping?",

#     # Cross-paper question — tests if retrieval finds the right paper
#     "What deep learning architectures did you use across your research?",

#     # Impossible question — tests the "I don't know" behaviour
#     "What is the boiling point of water?",
# ]

# for question in test_questions:
#     result = ask(question, verbose=True)
#     input("\nPress Enter for next question...")  # pause so you can read


file_title_dict = {'emg_speech_2025.pdf': 'Confidence-Based Self-Training for EMG-to-Speech: Leveraging Synthetic EMG for Robust Modeling', 
                 'icann_2024.pdf': 'Developmental Predictive Coding Model for Early Infancy Mono and Bilingual Vocal Continual Learning', 
                 'icdl_2022.pdf': 'Visuo-Motor Remapping for 3D, 6D and Tool-Use Reach using Gain-Field Networks', 
                 'neural_networks_2026.pdf': 'Structure from rank: Rank-order coding as a bridge from sequence to structure'}

file_to_title = {}
title_to_file = {}
for doc in vectorstore.docstore._dict.values():
    filename = doc.metadata.get("filename")
    title = file_title_dict[filename]
    
    file_to_title[filename] = title
    title_to_file[title] = filename



# Component 7 — Gradio UI (for Gradio 6.9.0)

import gradio as gr

print("\n=== Launching Gradio UI ===")
print(f"Gradio version: {gr.__version__}")


def gradio_ask(question, paper_name):
    file_name = title_to_file[paper_name]
    """Wrapper for Gradio — returns (answer_text, sources_text)"""
    if not question or not question.strip():
        return "Please enter a question.", ""

    try:
        result = ask(question, file_name, verbose=False)

        # Format sources for display
        seen = set()
        sources_lines = []
        for doc in result["sources"]:
            # Safely get source and page
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            source = metadata.get("source", "unknown")
            src = os.path.basename(source)
            page = metadata.get("page", "?")
            display = metadata.get("display", src)
            
            key = f"{display}:p{page}"
            if key not in seen:
                # Show source + a short excerpt
                excerpt = doc.page_content[:150].replace("\n", " ")
                sources_lines.append(f"📄 {display}  (page {page})\n   \"{excerpt}...\"")
                seen.add(key)

        sources_text = "\n\n".join(sources_lines) if sources_lines else "No sources retrieved."
        answer = result["answer"] if sources_lines else "The question and the selected paper are unmatched."
        return answer, sources_text
        
    except Exception as e:
        print(f"Error in gradio_ask: {e}")
        import traceback
        traceback.print_exc()
        return f"Error: {str(e)}", "Check console for details"

# For Gradio 6.x, use the new API
demo = gr.Interface(
    fn=gradio_ask,
    
    # In Gradio 6.x, use gr.Textbox directly (not gr.inputs.Textbox)
    inputs=[gr.Textbox(
        label="Ask about my research",
        placeholder="e.g., What is rank-order coding and how does it relate to neural sequences?",
        lines=2,
        ),
        gr.Dropdown(
            choices=title_to_file,
            label="Select Paper",
        )],
    
    # In Gradio 6.x, outputs are also direct components
    outputs=[
        gr.Textbox(label="Answer", lines=8),
        gr.Textbox(label="Retrieved from", lines=6),
    ],
    
    title="Chat with my PhD Research",
    description="Ask questions about my published papers. Answers are grounded in the actual paper text.",
    
    examples=[
        ["What dataset was used for the EMG-to-speech experiments?"],
        ["How does the EMG-to-speech model work?"],
        ["How were gain-field networks used for visuo-motor remapping?"],
        ["What is rank-order coding and how does it relate to neural sequences?"]
    ],
    
    # In Gradio 6.x, allow_flagging is renamed to 'flagging_mode'
    flagging_mode="manual",
    flagging_dir="./flagged",
)

# Launch the interface
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
)