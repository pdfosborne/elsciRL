import ollama
from typing import Optional, Dict, Any
import json
import logging

class LLMInstructionValidator:
    """
    A class for validating if a given text matches or completes an instruction
    using Large Language Model reasoning via Ollama.
    """
    
    def __init__(self, 
                 model: str = "llama3.2"):
        """
        Initialize the LLM Instruction Validator.
        
        Args:
            model: Ollama model to use for validation (e.g., "llama3.2", "mistral", "codellama").
            temperature: Temperature for LLM responses (lower = more deterministic).
            host: Ollama host URL. If None, uses default localhost.
        """
        self.model = model
        self.logger = logging.getLogger(__name__)
    
    def validate_instruction_completion(self, 
                                      instruction_description: str, 
                                      best_match: str) -> Dict[str, Any]:
        """
        Compare instruction description with best match to determine if the 
        best match completes or fulfills the instruction.
        
        Args:
            instruction_description: The original instruction or task description
            best_match: The text/response that potentially completes the instruction
            
        Returns:
            Dict containing:
                - 'is_complete': Boolean indicating if instruction is completed
                - 'confidence': Float between 0-1 indicating confidence level
                - 'reasoning': String explaining the LLM's reasoning
                - 'partial_completion': Boolean if partially completed
        """
        
        # Construct the prompt for the LLM
        prompt = self._construct_validation_prompt(instruction_description, best_match)
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at evaluating whether responses complete given instructions. "
                                 "You must respond with valid JSON format."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                options={
                    "num_predict": 500
                }
            )
            
            # Parse the LLM response
            result = self._parse_llm_response(response['message']['content'])
            
            self.logger.info(f"Validation completed. Is complete: {result['is_complete']}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during LLM validation: {str(e)}")
            return {
                'is_complete': False,
                'confidence': 0.0,
                'reasoning': f"Error occurred during validation: {str(e)}",
                'partial_completion': False
            }
    
    def _construct_validation_prompt(self, instruction: str, match: str) -> str:
        """
        Construct the prompt for the LLM to evaluate instruction completion.
        
        Args:
            instruction: The instruction description
            match: The best match text
            
        Returns:
            Formatted prompt string
        """
        prompt = f"""
Please evaluate whether the "Best Match" text completes or fulfills the given "Instruction".
You only need to confirm that the language match well and do not need to check if the best match would update the environment.
The language structure from the environment is fixed and does not change, so do not expect a 'Best Match' that is better structured than what is given.
You need to determine if the current 'Best Match' is likely to be the best match for the instruction given the language structure of the environment, do not expect more detail than what is given.

INSTRUCTION:
{instruction}

BEST MATCH:
{match}

Analyze if the Best Match adequately completes, addresses, or fulfills the Instruction. Consider:
1. Does it directly address what was asked?
2. Is the response complete and comprehensive?
3. Does it meet the intent of the instruction?

Respond ONLY with valid JSON in this exact format:
{{
    "is_complete": true/false,
    "confidence": 0.0-1.0,
    "reasoning": "Brief explanation of your evaluation",
    "partial_completion": true/false
}}

Your confidence should be:
- 0.9-1.0: Very confident the instruction is completed
- 0.7-0.8: Mostly confident but some minor gaps
- 0.5-0.6: Partially completed with significant gaps
- 0.0-0.4: Does not complete the instruction
"""
        return prompt
    
    def _parse_llm_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse the LLM response and extract validation results.
        
        Args:
            response_text: Raw text response from LLM
            
        Returns:
            Parsed validation results
        """
        try:
            # Clean and parse JSON response
            cleaned_response = response_text.strip()
            
            # Handle potential markdown code blocks
            if "```json" in cleaned_response:
                start = cleaned_response.find("```json") + 7
                end = cleaned_response.find("```", start)
                cleaned_response = cleaned_response[start:end].strip()
            elif "```" in cleaned_response:
                start = cleaned_response.find("```") + 3
                end = cleaned_response.find("```", start)
                cleaned_response = cleaned_response[start:end].strip()
            
            result = json.loads(cleaned_response)
            
            # Validate required fields
            required_fields = ['is_complete', 'confidence', 'reasoning', 'partial_completion']
            for field in required_fields:
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
            
            # Ensure confidence is between 0 and 1
            result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))
            
            return result
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.error(f"Failed to parse LLM response: {str(e)}")
            return {
                'is_complete': False,
                'confidence': 0.0,
                'reasoning': f"Failed to parse LLM response: {response_text[:100]}...",
                'partial_completion': False
            }
    
    def batch_validate(self, instruction_match_pairs: list) -> list:
        """
        Validate multiple instruction-match pairs in batch.
        
        Args:
            instruction_match_pairs: List of tuples (instruction, best_match)
            
        Returns:
            List of validation results
        """
        results = []
        for instruction, match in instruction_match_pairs:
            result = self.validate_instruction_completion(instruction, match)
            results.append(result)
        
        return results
    
    def list_available_models(self) -> list:
        """
        List available Ollama models.
        
        Returns:
            List of available model names
        """
        try:
            models = ollama.list()
            return [model['model'].split(':')[0] for model in models['models']]
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            return []


# Example usage and convenience function
def validate_instruction_match(instruction_description: str, 
                             best_match: str,
                             model: str = "llama3.2") -> Dict[str, Any]:
    """
    Convenience function to quickly validate if a best match completes an instruction.
    
    Args:
        instruction_description: The instruction to validate against
        best_match: The text that potentially completes the instruction
        model: Ollama model to use (default: "llama2")
        host: Ollama host URL (optional)
        
    Returns:
        Validation result dictionary
    """
    validator = LLMInstructionValidator(model=model)
    return validator.validate_instruction_completion(instruction_description, best_match)
