The following document and code are done 
by Vernon and Jasmine

Required installation
pip install --upgrade pip
pip install langchain-ollama
pip install chromadb sentence-transformers rank_bm25
pip install pypdf
python -m pip install langchain langchain-community langchain-core
pip install gradio
pip install langchain-openai

Recommended installation
pip install langchainhub

For the python version it would is recommended to have it <3.10

Take note for the Data files that would be used for this Local LLM would be from the following sites:
https://hpp.moh.gov.sg/guidelines/

run: python app.py

Role-based for staff
Username: admin
Password: clinical123

note for metric testing files:
the evaluation scripts are organized in the metric_tests/ directory for repository organization. However, all experiments were conducted locally from the project's root directory (RAGML), alongside app.py, ingest.py, retriever.py, and the Data/ folder.
For additional references and roadmap of the overall architecture layout, the Full_Architecture_Modal.png will show both the full layout and the evaluation layout
