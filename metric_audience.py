import csv
import json
import re
import shutil
import statistics
import time
from pathlib import Path

from ingest import load_and_chunk_documents
from retriever import get_hybrid_retriever
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

DATA_DIR = './data'  # Use './Data' if that is the actual directory.
EVAL_DB = Path('./chroma_audience_staff_public_eval')
MODEL = 'gemma4:e2b'
K = 4

TOPICS = [
    ('hand_hygiene', 'When should hand hygiene be performed during patient care in a hospital?'),
    ('ppe', 'What protective equipment is used for contact precautions in hospital care?'),
    ('standard_precautions', 'What are standard precautions in infection prevention and control?'),
    ('respiratory_hygiene', 'What respiratory hygiene measures should be followed in a hospital?'),
    ('environmental_cleaning', 'How should frequently touched surfaces be cleaned in hospital patient-care areas?'),
]

STAFF_PROMPT = ChatPromptTemplate.from_template('''Answer strictly using context:
{context}

Question: {question}
Answer with full clinical technical details, source citation, and the relevance score percentage:''')
PUBLIC_PROMPT = ChatPromptTemplate.from_template('''Answer strictly using context:
{context}

Question: {question}
Answer in plain, easy-to-understand language for patients, include source citation and the relevance score percentage:''')

JUDGE_PROMPT = ChatPromptTemplate.from_template('''You are an evaluator of a clinical information assistant.
Evaluate communication suitability for the intended audience, not independent medical accuracy.
Audience: {audience}
Question: {question}
Response: {answer}

For each criterion assign integer 0 (poor), 1 (partial), or 2 (good):
- appropriateness: level of information suitable for the audience
- clarity: comprehensible to the audience
- terminology: technical terms appropriately used or explained
- actionability: useful information within the audience's role
A refusal or empty answer should have actionability 0. Do not reward missing information.
Return ONLY JSON with exactly these fields:
{{"appropriateness":0,"clarity":0,"terminology":0,"actionability":0,"reason":"short rationale"}}''')
KEYS = ('appropriateness', 'clarity', 'terminology', 'actionability')


def text_of(response):
    value = response.content
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(x if isinstance(x, str) else str(x.get('text', '')) for x in value)
    return str(value)


def score_answer(judge, audience, question, answer):
    if not answer.strip():
        return None, 'Empty response'
    for attempt in range(2):
        try:
            raw = text_of((JUDGE_PROMPT | judge).invoke({
                'audience': audience, 'question': question, 'answer': answer
            }))
            match = re.search(r'\{.*\}', raw, re.S)
            if not match:
                raise ValueError('No JSON object in evaluator output')
            data = json.loads(match.group())
            if any(type(data.get(k)) is not int or data[k] not in (0, 1, 2) for k in KEYS):
                raise ValueError('Invalid evaluator score')
            return data, ''
        except Exception as exc:
            error = str(exc)
    return None, error


def write_csv(path, rows, fields):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not Path(DATA_DIR).is_dir():
        raise FileNotFoundError(f'Cannot find {DATA_DIR}. Change DATA_DIR if necessary.')
    chunks = load_and_chunk_documents(DATA_DIR)
    if not chunks:
        raise RuntimeError('No PDF chunks found')
    if EVAL_DB.exists():
        shutil.rmtree(EVAL_DB)  # Dedicated test index only.
    _, vectorstore = get_hybrid_retriever(chunks, persist_dir=str(EVAL_DB))
    generator = ChatOllama(model=MODEL, temperature=0)
    judge = ChatOllama(model=MODEL, temperature=0)
    generator.invoke('Reply OK')
    judge.invoke('Reply OK')

    fields = ['topic', 'audience', 'question', 'answer', 'retrieved_context',
              'retrieved_chunks', 'retrieval_seconds', 'generation_seconds',
              'judge_seconds', *KEYS, 'suitability_percent', 'judge_reason',
              'judge_error', 'human_verified']
    rows = []
    for topic, question in TOPICS:
        for audience in ('staff', 'public'):
            # This reproduces the intended role access filter in app.py.
            role_filter = ({'access_level': 'public'} if audience == 'public'
                           else {'access_level': {'$in': ['public', 'staff']}})
            t0 = time.perf_counter()
            docs_scores = vectorstore.similarity_search_with_score(
                question, k=K, filter=role_filter
            )
            retrieval_seconds = time.perf_counter() - t0
            context_parts = []
            for doc, distance in docs_scores:
                score = max(0.0, min(1.0, 1.0 - distance / 2.0))
                context_parts.append(
                    f"[Doc: {doc.metadata.get('source', 'SOP')} | "
                    f"Page: {int(doc.metadata.get('page', 0)) + 1} | "
                    f"Score: {score:.2%}]\n{doc.page_content}"
                )
            context = '\n\n'.join(context_parts)
            if not context:
                # No public evidence is not the same as a bad public-language answer.
                answer = 'Insufficient accessible guideline information to answer this question.'
                generation_seconds = 0.0
            else:
                prompt = STAFF_PROMPT if audience == 'staff' else PUBLIC_PROMPT
                t0 = time.perf_counter()
                try:
                    answer = text_of((prompt | generator).invoke({
                        'context': context, 'question': question
                    }))
                except Exception as exc:
                    answer = ''
                    print('Generation failed:', exc)
                generation_seconds = time.perf_counter() - t0
            t0 = time.perf_counter()
            if not context:
                scores, error = None, 'No accessible documents; suitability not scored'
            else:
                scores, error = score_answer(judge, audience, question, answer)
            judge_seconds = time.perf_counter() - t0
            row = dict(topic=topic, audience=audience, question=question,
                       answer=answer, retrieved_context=context,
                       retrieved_chunks=len(docs_scores),
                       retrieval_seconds=round(retrieval_seconds, 3),
                       generation_seconds=round(generation_seconds, 3),
                       judge_seconds=round(judge_seconds, 3),
                       suitability_percent='', judge_reason=scores.get('reason', '') if scores else '',
                       judge_error=error, human_verified='')
            for k in KEYS:
                row[k] = scores[k] if scores else ''
            if scores:
                row['suitability_percent'] = round(sum(scores[k] for k in KEYS) / 8 * 100, 1)
            rows.append(row)
            write_csv('audience_results.csv', rows, fields)
            print(f'{topic} | {audience} | retrieved={len(docs_scores)} | score={row["suitability_percent"]}', flush=True)

    summary = []
    for audience in ('staff', 'public'):
        group = [r for r in rows if r['audience'] == audience]
        valid = [r['suitability_percent'] for r in group if r['suitability_percent'] != '']
        summary.append(dict(audience=audience, questions=len(group),
                            answers_scored=len(valid),
                            questions_with_accessible_context=sum(r['retrieved_chunks'] > 0 for r in group),
                            average_suitability_percent=round(statistics.mean(valid), 1) if valid else ''))
    write_csv('audience_summary.csv', summary, list(summary[0]))
    print('Saved audience_results.csv and audience_summary.csv')
    print('Scores are preliminary LLM judgments. No-context answers are excluded, not scored zero.')


if __name__ == '__main__':
    main()
