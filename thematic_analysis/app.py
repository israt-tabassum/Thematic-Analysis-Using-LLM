from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import requests
import json
import re
import nltk
import random
from nltk.corpus import wordnet
import os

nltk.download("wordnet")
nltk.download("omw-1.4")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "uploads"
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# --------------Text processing functions-----------------------

def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z/\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def synonym_augmentation(text, replace_prob=0.15):
    words = text.split()
    new_words = []
    for w in words:
        if random.random() < replace_prob and len(w) > 3:
            synonyms = set()
            for syn in wordnet.synsets(w):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace("_", " ")
                    if synonym != w:
                        synonyms.add(synonym)
            if synonyms:
                new_words.append(random.choice(list(synonyms)))
            else:
                new_words.append(w)
        else:
            new_words.append(w)
    return " ".join(new_words)

def synonym_replacement(text, n=2):
    words = text.split()
    if len(words) < 2: 
        return text
    new_words = words.copy()
    candidates = [w for w in words if len(w) > 3]
    random.shuffle(candidates)
    replaced = 0
    for word in candidates:
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace("_", " ")
                if synonym != word:
                    synonyms.add(synonym)
        if synonyms:
            new_words = [
                random.choice(list(synonyms)) if w == word else w
                for w in new_words
            ]
            replaced += 1
        if replaced >= n:
            break
    return " ".join(new_words)

def template_expand(text):
    templates = [
        text,
        f"this helps with {text}",
        f"support related to {text}",
        f"useful for {text}",
        f"important for {text}"
    ]
    return random.choice(templates)

def augment_text(text, num_syn_replacements=5):
    augmented = [text]
    for _ in range(num_syn_replacements):
        augmented.append(synonym_replacement(text, n=5))
    augmented.append(synonym_augmentation(text))
    augmented.append(synonym_augmentation(text, replace_prob=0.35))
    augmented.append(synonym_augmentation(text, replace_prob=0.05))
    augmented.append(synonym_augmentation(text, replace_prob=0.25))
    augmented.append(template_expand(text))
    return list(set(augmented))

def get_text_column(df):
    for col in df.columns:
        if df[col].dtype == object:
            return col
    raise ValueError("No text column found.")


# --------------------LLM API call------------------------------

def call_llm(prompt, model):
    LLM_URL = "http://localhost:11434/api/generate" 
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0}
    }
    r = requests.post(LLM_URL, json=payload)
    r.raise_for_status()
    return r.json()["response"]

def extract_json_safely(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
    raise ValueError("Invalid JSON from LLM")

def generate_themes_with_intent(group_name, original_texts, augmented_texts, model):
    prompt = f"""
STRICT INSTRUCTIONS:
- Output VALID JSON ONLY
- No explanations, no markdown
- Generate parent themes and subthemes for {group_name} automatically from these items.
- Parent themes should be broader, not too specific.
- Do NOT modify or remove any original dataset items in the subthemes.
- All data should be in the subthemes (total number of data and total number of subtheme should be same).
- For each parent theme, also generate an "intent" that summarizes the purpose of this theme and its subthemes.
- Use the augmented data for understanding context, relationships, and concept coverage, but generate themes and subthemes ONLY from the ORIGINAL dataset items.
- DO NOT USE AUGMENTED DATA as SUBTHEMES
- Subthemes MUST be exactly the input ORIGINAL dataset items.

Format:
{{
  "themes": [
    {{
      "theme_name": "",
      "intent": "",
      "subthemes": []
    }}
  ]
}}

ORIGINAL DATASET ITEMS:
{original_texts}

AUGMENTED DATA (for context only):
{augmented_texts}
"""
    response = call_llm(prompt, model)
    return extract_json_safely(response)

def flatten_subthemes_with_intent(theme_group):
    flattened = []
    for t in theme_group["themes"]:
        subthemes = t.get("subthemes", [])
        t["subthemes"] = [str(s) for s in subthemes]
        t["intent"] = str(t.get("intent", ""))
        flattened.append(t)
    return {"themes": flattened}


# ---------------------Flask routes--------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files.get("file")
        model = request.form.get("llm_model")
        if not file or not model:
            return "Please upload a file and select an LLM model."

        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

       
        df = pd.read_excel(filepath)
        df = df.drop_duplicates()


        text_col = get_text_column(df)
        texts = df[text_col].dropna().astype(str).tolist()


        clean_texts = [normalize_text(t) for t in texts]
        augmented_texts = []
        for t in clean_texts:
            augmented_texts.extend(augment_text(t))


        themes = generate_themes_with_intent("Dataset", clean_texts, augmented_texts, model)
        themes_flat = flatten_subthemes_with_intent(themes)

        return render_template("results.html", themes=themes_flat)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)

