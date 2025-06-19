import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask, request, render_template
from huggingface_hub import login
from llama_index.core import Settings
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from prometheus_flask_exporter import PrometheusMetrics
import mlflow
from datetime import datetime


app = Flask(__name__)
metrics = PrometheusMetrics(app, path="/metrics")

# MLflow Setup
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment("PDF_QA_Experiment")

load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

#'your_huggingface_token' with your personal huggingface token (is free to get it)


# LlamaIndex Settings
Settings.llm = HuggingFaceLLM(
    model_name="meta-llama/Llama-3.2-1B-Instruct",
    tokenizer_name="meta-llama/Llama-3.2-1B-Instruct",
    device_map="auto",
    context_window=2048,
    max_new_tokens=128,
    generate_kwargs={"temperature": 0.2},
)

Settings.embed_model = HuggingFaceEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
Settings.node_parser = SentenceSplitter(chunk_size=1000, chunk_overlap=50)

ALLOWED_EXTENSIONS = {'pdf'}
cache = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def build_query_engine_from_file(file_stream, filename):
    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_path = os.path.join(tmpdirname, filename)
        file_stream.seek(0)
        with open(tmp_path, 'wb') as f:
            f.write(file_stream.read())
        docs = SimpleDirectoryReader(tmpdirname).load_data()
        index = VectorStoreIndex.from_documents(docs)
        return index.as_query_engine()

@app.route('/', methods=['GET', 'POST'])
def index():
    answer = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file part in the request", 400
        file = request.files['file']
        question = request.form.get('question', '').strip()

        if file.filename == '':
            return "No file selected", 400
        if not allowed_file(file.filename):
            return "File type not allowed. Please upload a PDF.", 400
        if question == '':
            return "Please enter a question.", 400

        filename = Path(file.filename).name

        if filename not in cache:
            cache[filename] = build_query_engine_from_file(file.stream, filename)
        else:
            file.stream.seek(0)

        start_time = datetime.now()
        answer = str(cache[filename].query(question))
        end_time = datetime.now()

        with mlflow.start_run():
            mlflow.log_param("filename", filename)
            mlflow.log_param("question", question)
            mlflow.log_metric("response_time", (end_time - start_time).total_seconds())
            mlflow.log_text(answer, "answer.txt")

    return render_template('index.html', answer=answer)


if __name__ == '__main__':
    print("\nRegistered Routes:")
    for rule in app.url_map.iter_rules():
        print(f"{rule.endpoint:25s} -> {rule.rule}")
    app.run(host='0.0.0.0', port=8000, debug=False)
