import os
import fitz  # PyMuPDF
from flask import Flask, request, render_template, jsonify
import openai
import asyncio
import time

openai.api_key = os.environ.get('OPENAI_API_KEY', 'sk-your-default-key')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ''
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

async def extract_ners_from_text(text, perspective):
    prompt = (
        f"Please extract the following named entities from the given {perspective}. Ensure the data is accurate to 10 decimal places "
        f"and provide the entities in a clear, consistent format:\n\n"
        f"1. Role\n"
        f"2. Experience\n"
        f"3. Work Experience\n"
        f"4. Education\n\n"
        f"Text:\n{text}\n\n"
        f"Extracted Named Entities:"
    )

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=200
    )
    content = response.choices[0].message['content'].strip()
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    return content, prompt_tokens, completion_tokens

async def compare_ners(resume_ners, jd_ners):
    comparison_prompt = (
        f"Compare the following named entities extracted from a candidate's resume and a company's job description. "
        f"Provide a similarity score accurate to 10 decimal places. Consider the relevance, context, and importance of each entity "
        f"in the context of job matching. Assign a similarity score between -1.0000000000 and 1.0000000000, with -1.0000000000 indicating no match, 0.0000000000 indicating neutral, "
        f"and 1.0000000000 indicating a perfect match. Give extra importance to the Role and relevant first-level skills/experience match. "
        f"Use the following weights: 50% for Work Experience and Role, 35% for first-level Skills, and 15% for Education. "
        f"Do not compare second and third-level skills. Follow these specific instructions for accuracy and format:\n\n"
        f"1. Role: Compare the role title and description in both the resume and job description.\n"
        f"2. Experience: Evaluate the relevance and length of experience mentioned.\n"
        f"3. Work Experience: Assess the alignment of past job responsibilities and achievements.\n"
        f"4. Education: Compare educational qualifications and their relevance to the job role.\n\n"
        f"Resume Named Entities:\n{resume_ners}\n\n"
        f"Job Description Named Entities:\n{jd_ners}\n\n"
        f"Output the similarity score followed by a concise summary of the reasoning in the exact format below:\n"
        f"Similarity Score: <score>\n"
        f"Reasoning: <concise summary of reasoning>"
    )

    response = await openai.ChatCompletion.acreate(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": comparison_prompt}
        ],
        max_tokens=200
    )

    result = response.choices[0].message['content'].strip()
    prompt_tokens = response['usage']['prompt_tokens']
    completion_tokens = response['usage']['completion_tokens']
    try:
        score_part, reasoning_part = result.split('Reasoning:', 1)
        score = score_part.split(':')[1].strip()
        reasoning = reasoning_part.strip()
    except ValueError:
        score = "N/A"
        reasoning = "Unable to determine similarity score and reasoning from the response."
    return score, reasoning, prompt_tokens, completion_tokens

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/compare', methods=['POST'])
def compare():
    start_time = time.time()
    
    jd_text = request.form['jd_text']
    resume_file = request.files['resume_file']

    if resume_file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], resume_file.filename)
        resume_file.save(filename)

        resume_text = extract_text_from_pdf(filename)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        jd_ners, jd_prompt_tokens, jd_completion_tokens = loop.run_until_complete(extract_ners_from_text(jd_text, "job description"))
        resume_ners, resume_prompt_tokens, resume_completion_tokens = loop.run_until_complete(extract_ners_from_text(resume_text, "resume"))

        similarity_score, reasoning, comparison_prompt_tokens, comparison_completion_tokens = loop.run_until_complete(compare_ners(resume_ners, jd_ners))

        total_prompt_tokens = jd_prompt_tokens + resume_prompt_tokens + comparison_prompt_tokens
        total_completion_tokens = jd_completion_tokens + resume_completion_tokens + comparison_completion_tokens

        # Calculate costs
        prompt_cost = total_prompt_tokens * 0.50 / 1_000_000
        completion_cost = total_completion_tokens * 1.50 / 1_000_000
        total_cost = prompt_cost + completion_cost

        end_time = time.time()
        total_time = end_time - start_time

        return render_template('result.html', score=similarity_score, reasoning=reasoning, 
                               total_cost=total_cost, total_time=total_time)
    else:
        return "No file uploaded", 400

@app.route('/api/compare', methods=['POST'])
def api_compare():
    start_time = time.time()
    
    data = request.json
    jd_text = data['jd_text']
    resume_text = data['resume_text']

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    jd_ners, jd_prompt_tokens, jd_completion_tokens = loop.run_until_complete(extract_ners_from_text(jd_text, "job description"))
    resume_ners, resume_prompt_tokens, resume_completion_tokens = loop.run_until_complete(extract_ners_from_text(resume_text, "resume"))

    similarity_score, reasoning, comparison_prompt_tokens, comparison_completion_tokens = loop.run_until_complete(compare_ners(resume_ners, jd_ners))

    total_prompt_tokens = jd_prompt_tokens + resume_prompt_tokens + comparison_prompt_tokens
    total_completion_tokens = jd_completion_tokens + resume_completion_tokens + comparison_completion_tokens

    # Calculate costs
    prompt_cost = total_prompt_tokens * 0.50 / 1_000_000
    completion_cost = total_completion_tokens * 1.50 / 1_000_000
    total_cost = prompt_cost + completion_cost

    end_time = time.time()
    total_time = end_time - start_time

    return jsonify({
        'score': similarity_score,
        'reasoning': reasoning,
        'total_cost': total_cost,
        'total_time': total_time
    })

if __name__ == '__main__':
    app.run(debug=True)
