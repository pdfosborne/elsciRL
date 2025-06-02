import logging
import ollama
from elsciRL.adapters.LLM_logic_generators.adapter_prompt import adapter_prompt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OllamaAdapterGenerator:
    def __init__(self, pseudocode_model: str, save_pseudocode: bool = False, pseudocode_file_path: str = None):
        """
        Initializes the OllamaAdapterGenerator.

        Args:
            primary_ollama_model_func: A function that simulates/calls the first Ollama model 
                                       (equivalent to text_ollama with encode=False).
                                       It should take text and return transformed text.
            pseudocode_ollama_model_func: A function that simulates/calls the Ollama model 
                                          for generating pseudocode. It should take a prompt 
                                          and return the generated pseudocode string.
        """
        logging.info("OllamaAdapterGenerator initialized.")
        self.pseudocode_model = pseudocode_model
        if save_pseudocode:
            self.pseudocode_file_path = pseudocode_file_path
        else:
            self.pseudocode_file_path = None

    def _generate_pseudocode_via_ollama(self, prompt: str) -> str:
        """
        Simulates calling another Ollama model to generate pseudocode.
        """
        logging.info("Generating pseudocode via (simulated) Ollama LLM...")

        response = ollama.chat(
            model=self.pseudocode_model, # Or another model suitable for code generation
            messages=[
                {
                    'role': 'user',
                    'content': prompt,
                }
            ],
            stream=False
            )
        
        logging.info(f"Generated pseudocode (simulated):\n{prompt}")
        return response['message']['content']

    def generate_adapter_pseudocode(self, environment_states: dict, transformed_states: str) -> str:
        """
        Logs environment states, processes input text using a primary Ollama model (simulated),
        and then uses another Ollama model (simulated) to generate Python pseudocode
        for an adapter function.

        Args:
            environment_states: A dictionary representing states from the environment.
            transformed_states: The LLM generated states.

        Returns:
            A string containing Python-like pseudocode for the adapter function.
        """
        logging.info(f"Generating adapter pseudocode for input text: '{transformed_states}'")
        logging.info(f"Environment states: {environment_states}")


        # 1. Prepare data for the pseudocode-generating LLM
        prompt_for_pseudocode_llm = f"""
                Given the following information:
                1. Environment States: {environment_states}
                2. Transformed Output Text: "{transformed_states}"

                Generate Python-like pseudocode for an 'adapter_function' in the form defined by {adapter_prompt}.

                The pseudocode should outline the logic rules necessary to transform the original input text 
                (or a similar input) into the transformed output text, considering the environment states.
                The function should aim to replicate the transformation performed by the primary LLM.

                These logic rules can be defied directly by a set of functions such as:
                def adapter(state, legal_moves=[], episode_action_history=[], encode=True, indexed=False):
                    if state[0] == 'some_value':
                        return "{transformed_states[:0]}..." # (Adjust based on logic)
                    elif state[1] == 'some_other_value':
                        return "{transformed_states[:0]}..." # (Adjust based on logic)
                    else:
                        return "some_other_transformation..."

                Or a lookup dictionary or table such as:
                obs_mapping = {{
                    'some_value': 'some_other_value',
                    'some_other_value': 'some_other_other_value',
                    'some_other_other_value': 'some_other_other_other_value',
                }}
                def adapter(state, legal_moves=[], episode_action_history=[], encode=True, indexed=False):
                    return obs_mapping[state]

                The logic rules can use the current state, legal moves, and action history to determine the output.

                Please provide only the Python pseudocode for adapter_function.
            """
        logging.info("Constructed prompt for pseudocode generation LLM.")

        # 2. Pass data to another LLM to create Python pseudocode
        pseudocode = self._generate_pseudocode_via_ollama(prompt_for_pseudocode_llm)
        
        logging.info("Successfully generated adapter pseudocode.")
        if self.pseudocode_file_path:
            with open(self.pseudocode_file_path, 'w') as f:
                f.write(pseudocode)
            logging.info(f"Pseudocode saved to {self.pseudocode_file_path}")
        return pseudocode

if __name__ == '__main__':
    # Example Usage 
    # Initialize the generator
    adapter_gen = OllamaAdapterGenerator(pseudocode_model='llama3.2', save_pseudocode=True, pseudocode_file_path='./pseudocode_sample.py')

    # Example data
    sample_env_states = {'Location': 'London',
                         'Day': 'Monday', 
                         'Time': 'Morning', 
                         'Weather':{
                            "cloud_cover": "low",
                            "temperature": "70 degrees",
                            "humidity": "20%",
                            "wind_speed": "10 mph", 
                            "wind_direction": "N"
                            },
                        'Location': 'London',
                        'Day': 'Monday', 
                        'Time': 'Afternoon', 
                        'Weather':{
                            "cloud_cover": "moderate",
                            "temperature": "85 degrees",
                            "humidity": "40%",
                            "wind_speed": "15 mph", 
                            "wind_direction": "SW"
                            },
                        
                        }
    sample_output = ["The weather on Monday morning in London is sunny and dry, the temperature is 70 degrees and low humidity and a light breeze.",
                     "The weather on Monday afternoon in London is cloudy, the temperature is 85 degrees and moderate humidity and a moderate breeze from the south-west."]

    # Generate pseudocode
    generated_code = adapter_gen.generate_adapter_pseudocode(sample_env_states, sample_output)
    
