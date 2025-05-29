# IMPORTS LLM API TOOLS
# EDITED OUT FOR NOW UNTIL FULL IMPLEMENTATION READY

# import os
# import json

# from openai import OpenAI

# def call_gpt_api(prompt):
#     import os
#     api_key = os.environ.get("OPENAI_API_KEY")
#     client = OpenAI(api_key=api_key)
#     response = client.chat.completions.create(
#         model="gpt-4.1",
#         messages=[{"role": "system", "content": prompt}],
#         max_tokens=5000
#     )
#     return response.to_dict() if hasattr(response, 'to_dict') else response

# def process_gpt_response(response):
#     if response and 'choices' in response:
#         return response['choices'][0]['message']['content']
#     return None

# def generate_application(self, user_input:str=''):
#     # TODO: Use this in a new tab with user input to update application list
#     # Load the app_setup.md content as part of the system prompt
    
#     # Add requirement to system prompt for code chunk separation
#     system_prompt_requirement = (
#         "If your response contains any code chunks, you must output them in a separate section clearly marked as 'Code Output', "
#         "so that the application can extract and save them to a file. Do not mix code with explanations in the same section."
#     )
#     # Combine the app_setup.md info with the system prompt and the new requirement
#     system_prompt = (
#         "You are a helpful assistant. "
#         "Below is important application setup information for elsciRL:\n"
#         f"{self.app_setup_info}\n"
#         f"{system_prompt_requirement}\n"
#         "Please use this information to answer user queries."
#     )
    
#     if not user_input:
#         return {"error": "No input provided"}
    
#     # Use the utils function to call the GPT API
#     response = call_gpt_api(system_prompt + "\nUser: " + user_input)
#     reply = process_gpt_response(response)
#     print(reply)
#     if not reply:
#         return {"error": "Failed to get response from GPT API"}

#     # Save the complete output to a .txt file
#     output_dir = os.path.join(os.path.dirname(__file__), 'output')
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, 'last_gpt_response.txt')
#     with open(output_path, 'w', encoding='utf-8') as f:
#         f.write(str(response))

#     # Follow-up: Ask the AI model to extract all Python code and JSON config blocks and return a list of (filename, code) pairs
#     followup_prompt = (
#         "Extract all Python code blocks and JSON config blocks from the following text. "
#         "For each code or config block, output a JSON array where each item has 'filename' and 'code' fields. "
#         "Choose a descriptive filename for each code block (e.g., based on class/function names or comments, use .py for Python and .json for configs). "
#         "Do not include any explanation, only the JSON array.\n\n" + reply
#     )
#     code_response = call_gpt_api(followup_prompt)
#     code_reply = process_gpt_response(code_response)
#     try:
#         code_blocks = json.loads(code_reply)
#         generated_data = {}
#         for block in code_blocks:
#             fname = block.get('filename', 'extracted_code.py')
#             code = block.get('code', '')
#             generated_data[fname] = code
#             code_file_path = os.path.join(output_dir, fname)
#             with open(code_file_path, 'w', encoding='utf-8') as code_file:
#                 code_file.write(code)
#     except Exception as e:
#         # fallback: save the raw reply if not valid JSON
#         code_file_path = os.path.join(output_dir, 'extracted_code.py')
#         with open(code_file_path, 'w', encoding='utf-8') as code_file:
#             code_file.write(code_reply.strip())

#     for name,code in generated_data.items():
#         if 'engine' in name.lower():
#             generated_data['engine'] = code
#         elif 'analysis' in name.lower():
#             generated_data['analysis'] = code
#         elif ('experiment' in name.lower()) | ('agent' in name.lower()):
#             generated_data['agent_config'] = code
#         elif ('local' in name.lower()) | ('env' in name.lower()):
#             generated_data['local_config'] = code
#         elif 'adapter_language' in name.lower():
#             generated_data['adapter_language'] = code
#         elif ('numeric' in name.lower()) | ('default' in name.lower()):
#             generated_data['adapter_numeric'] = code
    
#     # Create the application setup dictionary
#     application_setup = {
#         'engine':generated_data['engine'],
#         'experiment_configs':{'quick_test':generated_data['agent_config']},
#         'local_configs':{'env_config':generated_data['local_config']},
#         'adapters':{'numeric_adapter':generated_data['adapter_numeric'],
#                     'language_adapter':generated_data['adapter_language']},
#         'local_analysis':{'blackjack_graphs':generated_data['analysis']},
#         'prerender_data':{},
#         'prerender_images':{},
#     }

#     # Add the new application to the application data
#     self.pull_app_data = self.application_data.add_applicaiton(
#         problem=generated_data['agent_config']['name'], 
#         application_data=application_setup
#         )

#     return reply