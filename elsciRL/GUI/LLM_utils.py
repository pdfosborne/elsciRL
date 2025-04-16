from openai import OpenAI

def call_gpt_api(prompt):
    import os
    api_key = os.environ.get("OPENAI_API_KEY")
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "system", "content": prompt}],
        max_tokens=5000
    )
    return response.to_dict() if hasattr(response, 'to_dict') else response

def process_gpt_response(response):
    if response and 'choices' in response:
        return response['choices'][0]['message']['content']
    return None