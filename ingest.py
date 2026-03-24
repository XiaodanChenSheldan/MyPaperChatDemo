# ingest.py
import os
import re
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["HF_HOME"] = "/data/xiaochen99/hf_cache"
os.environ["TRANSFORMERS_CACHE"] = "/data/xiaochen99/hf_cache"

import pymupdf4llm
import numpy as np
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

PDF_FOLDER = "./data"
INDEX_PATH = "./faiss_index"

# ─────────────────────────────────────────────
# STAGE 1 — EXTRACT CLEAN MARKDOWN
# ─────────────────────────────────────────────


def extract_markdown(pdf_path: str) -> str:
    """
    Convert PDF to markdown using pymupdf4llm.
    Handles double-column layout, figures, tables.
    """
    return pymupdf4llm.to_markdown(pdf_path, show_progress=False)

def remove_references_section(md_text: str) -> str:
    """
    Remove everything from References / Acknowledgment section onwards.
    Handles both markdown headers and plain text.
    """

    pattern = re.compile(
        r'\n\s*(?:#{1,6}\s*)?'  # optional markdown header
        r'(references|bibliography|works cited|literature|acknowledg(e)?ment[s]?)\b.*',
        re.IGNORECASE | re.DOTALL
    )

    match = pattern.search(md_text)
    if match:
        return md_text[:match.start()]

    return md_text

import re

def convert_markdown_tables(md_text: str) -> str:
    """
    Convert markdown tables into natural language sentences.
    Works for ANY table structure.
    """

    lines = md_text.split("\n")
    new_lines = []

    i = 0
    while i < len(lines):
        line = lines[i]

        # Detect table header (contains | and next line is separator)
        if "|" in line and i + 1 < len(lines) and re.match(r'^\s*\|?[-:\s|]+\|?\s*$', lines[i + 1]):
            header_line = line
            separator_line = lines[i + 1]

            headers = [h.strip() for h in header_line.strip("|").split("|")]

            i += 2  # skip header + separator

            # Process rows
            while i < len(lines) and "|" in lines[i]:
                row = [c.strip() for c in lines[i].strip("|").split("|")]

                if len(row) == len(headers):
                    sentence_parts = [
                        f"{headers[j]}: {row[j]}" for j in range(len(headers))
                    ]
                    new_lines.append(", ".join(sentence_parts) + ".")
                else:
                    new_lines.append(lines[i])  # fallback

                i += 1

            continue

        else:
            new_lines.append(line)
            i += 1

    return "\n".join(new_lines)


print("=== STAGE 1: Extracting markdown from PDFs ===\n")

raw_markdowns = {}   # filename → clean markdown text

for filename in sorted(os.listdir(PDF_FOLDER)):
    if not filename.endswith(".pdf"):
        continue

    filepath = os.path.join(PDF_FOLDER, filename)
    print(f"  Processing: {filename}")

    # Extract
    raw_md = extract_markdown(filepath)
    print(f"    → Raw markdown: {len(raw_md)} chars")

    # raw_md = convert_markdown_tables(raw_md)

    raw_md = remove_references_section(raw_md)

    raw_markdowns[filename] = raw_md

print(f"Total PDFs processed: {len(raw_markdowns)}")

# ─────────────────────────────────────────────
# STAGE 2 — CHUNKING
# ─────────────────────────────────────────────

print("\n=== STAGE 2: Splitting into chunks ===\n")

# Step A — split on markdown section headers first
# This gives semantically complete chunks:
# each chunk = one section of your paper
header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#",   "h1"),
        ("##",  "h2"),
        ("###", "h3"),
    ],
    strip_headers=False,
    # Keep the header text inside the chunk so the LLM
    # knows which section it's reading when answering
)

# Step B — if a section is still too long, split by characters
char_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=500,
    separators=["\n\n", "\n", ". ", " "],
)

all_chunks = []

for filename, md_text in raw_markdowns.items():
    print(f"  Chunking: {filename}")

    # Split by headers
    header_chunks = header_splitter.split_text(md_text)
    print(f"    → {len(header_chunks)} header-level chunks")

    file_chunks = []
    for hchunk in header_chunks:
        # Determine the section label from header metadata
        # MarkdownHeaderTextSplitter puts header text in metadata
        section = (
            hchunk.metadata.get("h3") or
            hchunk.metadata.get("h2") or
            hchunk.metadata.get("h1") or
            "Introduction"
        )

        if len(hchunk.page_content) <= 1000:
            # Small enough — keep as one chunk
            hchunk.metadata.update({
                "source":   f"./data/{filename}",
                "filename": filename,
                "section":  section,
                "display":  f"{filename} › {section[:50]}",
            })
            file_chunks.append(hchunk)
        else:
            # Too large — split further by characters
            sub_chunks = char_splitter.split_documents([hchunk])
            for i, sc in enumerate(sub_chunks):
                sc.metadata.update({
                    "source":   f"./data/{filename}",
                    "filename": filename,
                    "section":  section,
                    "display":  f"{filename} › {section[:50]} (part {i+1})",
                })
            file_chunks.extend(sub_chunks)

    print(f"    → {len(file_chunks)} final chunks after size splitting")

    # Show sample chunks so you can verify quality
    print(f"    Sample chunks:")
    for c in file_chunks[:3]:
        print(f"      section='{c.metadata['section'][:40]}'  "
              f"len={len(c.page_content)}  "
              f"text='{c.page_content[:80].replace(chr(10), ' ')}'...")

    all_chunks.extend(file_chunks)


print(f"\nTotal chunks across all papers: {len(all_chunks)}")
print(f"Average chunk size: {sum(len(c.page_content) for c in all_chunks) // len(all_chunks)} chars")

# ─────────────────────────────────────────────
# STAGE 3 — EMBED
# ─────────────────────────────────────────────

print("\n=== STAGE 3: Embedding chunks ===\n")

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda:0"},
    encode_kwargs={"normalize_embeddings": True},
)

# Quick similarity sanity check
print("Sanity check — embedding similarity:")
import numpy as np

def cosine_sim(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

v1 = embedding_model.embed_query("rank-order coding neural network")
v2 = embedding_model.embed_query("temporal spike sequence encoding")
v3 = embedding_model.embed_query("chocolate cake recipe")
print(f"  Related pair   : {cosine_sim(v1, v2):.3f}  (should be higher)")
print(f"  Unrelated pair : {cosine_sim(v1, v3):.3f}  (should be lower)")

print(f"\nEmbedding {len(all_chunks)} chunks...")
start = time.time()
vectorstore = FAISS.from_documents(all_chunks, embedding_model)
elapsed = time.time() - start
print(f"Done in {elapsed:.1f}s  ({len(all_chunks)/elapsed:.0f} chunks/sec)")

# ─────────────────────────────────────────────
# STAGE 4 — TEST RETRIEVAL BEFORE SAVING
# ─────────────────────────────────────────────

print("\n=== STAGE 4: Testing retrieval ===\n")

test_queries = [
    "How does the predictive coding model handle bilingual vocal learning?",
    "What dataset was used for EMG to speech conversion?",
    "What is rank-order coding and how does it relate to neural sequences?",
    "How were gain-field networks used for visuo-motor remapping?",
    "What deep learning architectures were used?",
]

all_good = True
for query in test_queries:
    results = vectorstore.similarity_search_with_score(query, k=2)
    print(f"Query: '{query}'")
    for doc, score in results:
        print(f"  score={score:.4f} | {doc.metadata['display']}")
        print(f"  text : {doc.page_content[:150].replace(chr(10), ' ')}...")

    # Flag if top result comes from wrong paper
    top_doc = results[0][0]
    print()

# ─────────────────────────────────────────────
# STAGE 5 — SAVE INDEX
# ─────────────────────────────────────────────

print("=== STAGE 5: Saving FAISS index ===\n")

vectorstore.save_local(INDEX_PATH)
print('*'*10)
print(vectorstore.docstore._dict[list(vectorstore.docstore._dict.keys())[0]].metadata)
for f in os.listdir(INDEX_PATH):
    size = os.path.getsize(os.path.join(INDEX_PATH, f))
    print(f"  {f}  ({size/1024:.1f} KB)")

print("\n=== INGESTION COMPLETE ===")
print(f"  PDFs processed : {len(raw_markdowns)}")
print(f"  Total chunks   : {len(all_chunks)}")
print(f"  Index saved to : {INDEX_PATH}/")
print("\nRun python app.py to start querying your research.")