import os
import json
import re
import google.generativeai as genai
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- 1. Configuration ---

try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    print("!!! ERROR: GEMINI_API_KEY environment variable not set.")

# Adjust model names if needed
generation_model = genai.GenerativeModel('gemini-2.0-flash')
analysis_model = genai.GenerativeModel('gemini-2.5-flash')

print("✅ Gemini Models configured.")


# --- 2. Core Toolkit Functions ---

def generate_text(prompt: str) -> str:
    response = generation_model.generate_content(prompt)
    return response.text


def analyze_full_report(prompt: str, output: str) -> dict:
    analysis_prompt = f"""
    You are a prompt analysis tool. Analyze the following pair.
    PROMPT: {prompt}
    OUTPUT: {output}
    Task 1: Word-by-Word Heatmap
    Go through the PROMPT word by word (including punctuation as tokens).
    Assign an "impact_score" (integer 1-5, 1=low, 5=high) to *every single word*.
    
    Task 2: Key Connections
    Identify the *key concepts* or *instructional words* from the PROMPT.
    For each key word, list the output phrases it influenced.

    Return *only* a valid JSON object in this format:
    {{
      "heatmap_data": [
        {{"word": "word1", "impact_score": 1}},
        ...
      ],
      "connections": [
        {{"prompt_word": "word2", "impact_score": 5, "influenced_output_words": ["output_phrase"]}}
        ...
      ]
    }}
    """
    try:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )
        response = analysis_model.generate_content(
            analysis_prompt,
            generation_config=generation_config
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"!!! ANALYSIS FAILED: {e}")
        return {
            "heatmap_data": [],
            "connections": [],
            "error": f"Error during analysis: {e}"
        }


def run_ablation_study(original_prompt: str, words_to_remove):
    new_prompt = original_prompt
    for word in words_to_remove:
        new_prompt = re.sub(
            r'\b' + re.escape(word) + r'\b',
            '',
            new_prompt,
            flags=re.IGNORECASE
        )
    new_prompt = re.sub(r'\s+', ' ', new_prompt).strip()
    return generate_text(new_prompt), new_prompt


def run_counterfactual_study(original_prompt: str, replacement_map: dict):
    new_prompt = original_prompt
    for old_word, new_word in replacement_map.items():
        new_prompt = re.sub(
            r'\b' + re.escape(old_word) + r'\b',
            new_word,
            new_prompt,
            flags=re.IGNORECASE
        )
    return generate_text(new_prompt), new_prompt


def quantify_change(original_output: str, new_output: str) -> dict:
    if original_output == new_output:
        return {"semantic_change_score": 1, "change_summary": "No change."}

    analysis_prompt = f"""
    You are an output analysis tool. Analyze the semantic difference between these two texts.
    **Original Output:** {original_output}
    ---
    **New Output:** {new_output}
    ---
    **Your Task:**
    1.  Assign a "semantic_change_score" (integer 1-10) based on how much the *core meaning, tone, or content* has changed.
    2.  Write a brief "change_summary" (one short sentence) explaining the most significant difference.
    Return *only* a valid JSON object: {{"semantic_change_score": <int>, "change_summary": "..."}}
    """
    try:
        generation_config = genai.GenerationConfig(
            response_mime_type="application/json"
        )
        response = analysis_model.generate_content(
            analysis_prompt,
            generation_config=generation_config
        )
        return json.loads(response.text)
    except Exception as e:
        return {
            "semantic_change_score": -1,
            "change_summary": f"Error during change analysis: {e}"
        }


# --- 3. API Endpoints ---

@app.route('/api/generate', methods=['POST'])
def http_generate():
    data = request.json or {}
    prompt = data.get('prompt')
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    try:
        output = generate_text(prompt)
        return jsonify({"output": output})
    except Exception as e:
        app.logger.exception("Error during generation")
        return jsonify({"error": f"Error during generation: {str(e)}"}), 500


@app.route('/api/analyze', methods=['POST'])
def http_analyze():
    data = request.json or {}
    prompt = data.get('prompt')
    output = data.get('output')
    if not prompt or not output:
        return jsonify({"error": "Prompt and output are required"}), 400

    analysis_data = analyze_full_report(prompt, output)
    return jsonify(analysis_data)


@app.route('/api/run_experiment', methods=['POST'])
def http_run_experiment():
    data = request.json or {}
    experiment_type = data.get('type')
    original_prompt = data.get('original_prompt')
    original_output = data.get('original_output')

    if not original_prompt or not original_output:
        return jsonify({"error": "original_prompt and original_output are required"}), 400

    if experiment_type == 'ablation':
        words_to_remove = data.get('changes', '').split(',')
        words_to_remove = [w.strip() for w in words_to_remove if w.strip()]
        new_output, new_prompt = run_ablation_study(original_prompt, words_to_remove)

    elif experiment_type == 'counterfactual':
        try:
            replacement_map = json.loads(data.get('changes', '{}'))
        except Exception:
            return jsonify({"error": "Invalid JSON for replacements"}), 400
        new_output, new_prompt = run_counterfactual_study(original_prompt, replacement_map)

    else:
        return jsonify({"error": "Invalid experiment type"}), 400

    new_analysis_data = analyze_full_report(new_prompt, new_output)
    change_data = quantify_change(original_output, new_output)

    return jsonify({
        "new_prompt": new_prompt,
        "new_output": new_output,
        "new_analysis": new_analysis_data,
        "change_data": change_data
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(debug=False, host='0.0.0.0', port=port)
