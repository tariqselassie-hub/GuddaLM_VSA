#!/usr/bin/env python3
import json
import requests
import os
import random
import time
import argparse

# Config
DATA_PATH = r"C:\Users\zoddj\zenith_research_divizion\GuddaLM\data\eclektic_main.jsonl"
OUT_PATH = "distillation_data.jsonl"
LLAMA_SERVER_URL = "http://127.0.0.1:8080/v1/completions"

def distill(num_samples=100):
    print(f"Loading seed prompts from {DATA_PATH}...")
    
    # Load dataset lines (extract some text to use as prompt)
    prompts = []
    if os.path.exists(DATA_PATH):
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    # Extract English content and question
                    context = data.get("en_c", "")
                    question = data.get("en_q", "")
                    if context and question:
                        prompts.append(f"Context: {context}\n\nQuestion: {question}\n\nAnswer:")
                except:
                    pass
    
    if not prompts:
        print("Fallback: Using synthetic prompts")
        prompts = [
            "Explain the theory of relativity.",
            "Write a short python script to reverse a string.",
            "What is the meaning of life?",
            "How does a CPU work?",
            "Describe the topology of a Heptal state."
        ]

    random.shuffle(prompts)
    prompts = prompts[:num_samples]

    print(f"Querying Gemma4 on llama.cpp server at {LLAMA_SERVER_URL} for {len(prompts)} samples...")
    
    with open(OUT_PATH, "w", encoding="utf-8") as out_f:
        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{len(prompts)}] Querying...")
            payload = {
                "prompt": prompt,
                "n_predict": 128,
                "temperature": 0.7,
                "stop": ["\n\n"]
            }
            
            try:
                resp = requests.post(LLAMA_SERVER_URL, json=payload, timeout=300)
                resp.raise_for_status()
                data = resp.json()
                completion = data["choices"][0]["text"].strip()
                
            except Exception as e:
                print(f"Error querying llama server ({e}). Falling back to synthetic completion.")
                completion = f"[Synthetic answer for: '{prompt[:30]}...'] - This is a mock completion because the local server at {LLAMA_SERVER_URL} could not be reached."
                
            # Save to JSONL
            record = {"prompt": prompt, "completion": completion}
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()
                
            time.sleep(0.1)
            
    print(f"Done. Saved to {OUT_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=50)
    args = parser.parse_args()
    distill(args.samples)
