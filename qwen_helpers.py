import transformers
from typing import Optional, List, Dict, Any, Tuple
import re

SYSTEM_PROMPT: str = (
    "Respond in the following format, potraying \"Apeiria\":\n"
    "[APEIRIA THINKS]\n"
    "<... thinking predure if requiring thinking ...>\n"
    "[APEIRIA SPEAKS]\n"
    "<... responses ...>"
) # FIXME: predure -> procedure

_WORD_RE = re.compile(r"\w+")
def _fast_word_count(text: str) -> int:
    # simple regex token count; ~10-50x faster than nltk.word_tokenize for our use
    return len(_WORD_RE.findall(text))

def apply_qwen_template(instruction: str, tokenizer: transformers.PreTrainedTokenizer, response: Optional[str]=None) -> Tuple[str, Optional[str]]:
    """Apply Qwen template to instruction"""
    system = "You're Apeiria, a world-class AI, investigating objects in a room."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
    ]
    instruction = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    if response is not None and "<|im_end|>" not in response:
        # If response does not end with <|im_end|>, add it
        response = response + "<|im_end|>" + tokenizer.eos_token

    return instruction, response

def remove_last_occurrence_rsplit(main_string, sub_string):
    """
    Removes the last occurrence of a substring from a string using rsplit() and join().

    Args:
        main_string: The string to modify.
        sub_string: The substring to remove.

    Returns:
        The modified string with the last occurrence of the substring removed,
        or the original string if the substring is not found.
    """
    parts = main_string.rsplit(sub_string, 1)
    
    if len(parts) > 1:
        return parts[0] + parts[1]
    else:
        # If the substring was not found, rsplit returns a list with the original string
        raise ValueError(f"Substring '{sub_string}' not found in the main string.")

def apply_qwen_template_with_partial_response(instruction: str, tokenizer: transformers.PreTrainedTokenizer, response: str) -> Tuple[str, str]:
    """Apply Qwen template to instruction with partial response"""
    system = "You're Apeiria, a world-class AI, investigating objects in a room."
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": instruction},
        # {"role": "assistant", "content": response},
    ]
    # enable_thinking=True, so that it doesn't add <think></think> as already outputted CoT in prompt
    instruction = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True) 
    # remove response end tokens - only the last one
    # response = remove_last_occurrence_rsplit(response, "<|im_end|>\n")
    instruction = instruction + response

    return instruction, response

def count_example_tokens(example, tokenizer):
    formatted_instruction, formatted_response = apply_qwen_template(example['description'], tokenizer, example['expected_response'])
    all_inputs = formatted_instruction + (formatted_response if formatted_response is not None else "")
    return len(tokenizer(all_inputs)['input_ids'])

def batch_apply_qwen_template(examples, tokenizer):
    """Batch apply Qwen template to multiple examples."""
    batch_messages = []
    responses = []
    for example in examples:
        # system = "You're Apeiria, a world-class AI, investigating objects in a room."
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example['description']},
        ]
        batch_messages.append(messages)
        responses.append(example['expected_response'])
    
    # Apply chat template to all examples
    formatted_instructions = tokenizer.apply_chat_template(
        batch_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Process responses
    processed_responses = []
    for response in responses:
        if response is not None:
            processed_response = response + "<|im_end|>" + tokenizer.eos_token
        else:
            processed_response = ""
        processed_responses.append(processed_response)
    
    # Combine instruction and response
    all_inputs = [instr + resp for instr, resp in zip(formatted_instructions, processed_responses)]
    
    return all_inputs

def batch_count_tokens(examples, tokenizer):
    """Count tokens for a batch of examples using batched tokenization."""
    all_inputs = batch_apply_qwen_template(examples, tokenizer)
    
    # Tokenize all inputs and get lengths
    tokenized = tokenizer(all_inputs, padding=False, truncation=False)
    lengths = [len(ids) for ids in tokenized['input_ids']]
    
    return {'token_count': lengths}

def batch_count_tokens_fast(examples, tokenizer):
    """Count tokens for a batch of examples using batched tokenization."""
    all_inputs = batch_apply_qwen_template(examples, tokenizer)
    
    # use regexp word matching to estimate token counts quickly
    lengths = [_fast_word_count(text) for text in all_inputs]
    
    return {'token_count': lengths}