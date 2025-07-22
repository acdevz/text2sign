import os
import ssl
import pprint
from flask import Flask, request, render_template, send_from_directory, jsonify
import stanza
from nltk.parse.stanford import StanfordParser
from nltk.tree import ParentedTree
from copy import deepcopy

# Allow unverified SSL for stanza
ssl._create_default_https_context = ssl._create_unverified_context

# Setup Flask
app = Flask(__name__, static_folder='static', static_url_path='')

# Setup Stanza
stanza.download('en', model_dir='stanza_resources')
en_nlp = stanza.Pipeline('en', processors={'tokenize': 'spacy'})

# Setup Stanford Parser
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
os.environ['CLASSPATH'] = os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17', 'stanford-parser.jar') + \
                          os.pathsep + os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17', 'stanford-parser-3.9.2-models.jar')
os.environ['STANFORD_MODELS'] = os.path.join(BASE_DIR, 'stanford-parser-full-2018-10-17')
parser = StanfordParser()

# Stopwords for ISL (static set)
STOP_WORDS = set(open("stopwords.txt").read().split())

# Globals for storing processed data
final_words_dict = {}

# --------------------------------------------------------
# Utils
# --------------------------------------------------------
def is_valid_word(word):
    valid_words = open("words.txt").read().splitlines()
    return word.lower() in valid_words

def letters_fallback(word):
    return list(word.upper())


# --------------------------------------------------------
# Sematic Simplification
# --------------------------------------------------------
# from transformers import T5ForConditionalGeneration, T5Tokenizer

# model_name = "t5-small"
# tokenizer = T5Tokenizer.from_pretrained(model_name)
# model = T5ForConditionalGeneration.from_pretrained(model_name)

# def simplify_text(text):
#     prompt = f"simplify: {text}"
#     inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
#     outputs = model.generate(inputs.input_ids, max_length=128, num_beams=4, early_stopping=True)
#     simplified = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     return simplified


# --------------------------------------------------------
# NLP Pipeline
# --------------------------------------------------------
def parse_and_reorder(sentence_tokens):
    parse_tree = list(parser.parse(sentence_tokens))[0]
    parent_tree = ParentedTree.convert(parse_tree)
    return reorder_tree_structure(parent_tree).leaves()

def reorder_tree_structure(tree):
    def label_subtrees(tree):
        return {sub.treeposition(): 0 for sub in tree.subtrees()}

    def handle_np(i, flag, mod_tree, sub):
        if flag[sub.treeposition()] == 0 and flag[sub.parent().treeposition()] == 0:
            flag[sub.treeposition()] = 1
            mod_tree.insert(i, sub)
            i += 1
        return i

    def handle_vp(i, flag, mod_tree, sub):
        for child in sub.subtrees():
            if child.label() in ('NP', 'PRP') and flag[child.treeposition()] == 0:
                flag[child.treeposition()] = 1
                mod_tree.insert(i, deepcopy(child))
                i += 1
        return i

    flag = label_subtrees(tree)
    mod_tree = ParentedTree('ROOT', [])
    i = 0
    for sub in tree.subtrees():
        if sub.label() == "NP":
            i = handle_np(i, flag, mod_tree, sub)
        elif sub.label() in ("VP", "PRP"):
            i = handle_vp(i, flag, mod_tree, sub)

    for sub in tree.subtrees():
        for child in sub.subtrees():
            if len(child.leaves()) == 1 and flag[child.treeposition()] == 0:
                flag[child.treeposition()] = 1
                mod_tree.insert(i, deepcopy(child))
                i += 1

    return mod_tree

# --------------------------------------------------------
# Glossification Process
# --------------------------------------------------------
def glossify(text):
    doc = en_nlp(text)
    final_sentences = []

    for sentence in doc.sentences:
        tokens = [w.text for w in sentence.words if w.text.lower() not in STOP_WORDS and w.upos != 'PUNCT']
        reordered = parse_and_reorder(tokens)
        lemmatized = [w.lemma for w in sentence.words if w.text in reordered]
        glossed = []
        for word in lemmatized:
            if is_valid_word(word):
                glossed.append(word.lower())
            else:
                glossed.extend(letters_fallback(word))
        final_sentences.append(glossed)

    return final_sentences

# --------------------------------------------------------
# Flask Routes
# --------------------------------------------------------
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def handle_post():
    text = request.form.get('text', '')
    if not text.strip():
        return jsonify({})

    final_words_dict.clear()
    # simplified_text = simplify_text(text)
    glossed_sentences = glossify(text)
    word_id = 1
    for sentence in glossed_sentences:
        for word in sentence:
            final_words_dict[word_id] = word.upper() if len(word) == 1 else word
            word_id += 1

    pprint.pprint(final_words_dict)
    return jsonify(final_words_dict)

@app.route('/static/<path:path>')
def serve_signfiles(path):
    return send_from_directory('static', path)

if __name__ == "__main__":
    app.run(host='0.0.0.0')
