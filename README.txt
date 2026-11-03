pip install --upgrade pip
pip install langchain-ollama
pip install chromadb sentence-transformers rank_bm25
pip install pypdf
python -m pip install langchain langchain-community langchain-core
pip install gradio

Recommended installation
pip install langchainhub

Take note for the Data files that would be used for this Local LLM would be from the following sites:
https://hpp.moh.gov.sg/guidelines/

run: python app.py

Role-based
Username: admin
Password: clinical123

note for metric testing files:
the evaluation scripts are organized in the metric_tests/ directory for repository organization. However, all experiments were conducted locally from the project's root directory (RAGML), alongside app.py, ingest.py, retriever.py, and the Data/ folder.
