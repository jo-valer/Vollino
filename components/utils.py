import os
import torch
import json
import re

from dotenv import load_dotenv
from argparse import Namespace
from typing import Tuple
from components.dst import DialogueState
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedTokenizer, PreTrainedModel


MODELS = {
    "llama2": "meta-llama/Llama-2-7b-chat-hf",
    "llama3": "meta-llama/Meta-Llama-3-8B-Instruct",
    "groq_llama3": "llama-3.1-8b-instant"
}


class LLMGenerator():
    """A class to prompt a LLM."""

    def __init__(self, dialogue_state: DialogueState, generate_fn, args):
        self.dialogue_state = dialogue_state
        self.generate_fn = generate_fn
        self.args = args

    def generate(self, system_prompt: str, user_prompt: str):
        """Generate a response from the LLM."""
        return self.generate_fn(
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict:
        """
        Generate a JSON response from the LLM. Try a second time, if the first time the generated text is not valid JSON.
        Silently fail on second attempt, returning None
        """
        response = self.generate(system_prompt, user_prompt)
        try:
            return extract_json(response)
        except ValueError as e:
            if self.args.debug: print(f"\033[91mError extracting JSON: {e}. Retrying...\033[0m")
            response = self.generate(system_prompt, user_prompt)
            try:
                return extract_json(response)
            except ValueError as e:
                if self.args.debug: print(f"\033[91mError extracting JSON on second attempt: {e}\033[0m")
                return None


def json_to_string(json_obj: dict, no_brackets=False) -> str:
    """Convert a JSON object to a string."""
    if no_brackets:
        return ', '.join(f'"{key}": "{value}"\n' for key, value in json_obj.items())
    return json.dumps(json_obj, indent=4)


def remove_nulls(obj: dict) -> dict:
    if isinstance(obj, dict):
        return {k: remove_nulls(v) for k, v in obj.items() if v is not None and v != "" and v != "null" and v != "None"}
    elif isinstance(obj, list):
        return [remove_nulls(v) for v in obj if v is not None and v != "" and v != "null" and v != "None"]
    else:
        return obj


def extract_json(text: str, remove_null: bool = True) -> dict:
    """
    Extract a json object from the text.
    Remove any characters before the json object.
    """
    try:
        # First of all, if 'False' or 'True' are found, lowercase
        text = text.replace('True', 'true').replace('False', 'false')
        # Find the start of the JSON object
        start = text.index('{')
        # Find the end of the JSON object
        end = text.rindex('}') + 1
        json_str = text[start:end]
        ret_json = json.loads(json_str)
    except (ValueError, json.JSONDecodeError) as e:
        try:
            json_str = text.strip()

            # Maybe there are multiple parentheses: {{...}}
            if json_str.startswith('{{') and json_str.endswith('}}'):
                json_str = json_str[1:-1].strip()

            # Maybe there are comments using hashtags
            pattern = r'(?<!["\'])#.*'
            cleaned = re.sub(pattern, '', json_str)

            # Maybe there is a ': None' (or else) in place of a ': null'
            cleaned = re.sub(r':\s*(None|none|Null|NULL)', ': null', cleaned)

            ret_json = json.loads(cleaned)
        except (AttributeError, json.JSONDecodeError) as e:
            raise ValueError(f"Could not extract JSON from text: {text}") from e
    if remove_null:
        ret_json = remove_nulls(ret_json)
    return ret_json


def set_file_name(filename: str, overwrite: bool = False) -> str:
    if not overwrite:
        i = 1
        new_filename = os.path.join(os.path.dirname(filename), str(i), os.path.basename(filename))
        while os.path.exists(new_filename) and not overwrite:
            i += 1
            new_filename = os.path.join(os.path.dirname(filename), str(i), os.path.basename(filename))
    return new_filename


def save_txt(filename: str, text: str, mode: str = "w"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, mode) as f:
        f.write(text)
        f.close()
    # print(f"Saved to {filename}")


def generate(
    generation_args: dict,
    system_prompt: str,
    user_prompt: str
) -> str:
    """
    Generate a response from the LLM.
    """
    if generation_args["model_type"] == "local":
        text = generation_args['args'].chat_template.format(system_prompt, user_prompt)
        inputs = generation_args['tokenizer'](text, return_tensors="pt").to(generation_args['model'].device)
        output = generation_args['model'].generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=generation_args['args'].max_new_tokens,
            pad_token_id=generation_args['tokenizer'].eos_token_id,
        )
        return generation_args['tokenizer'].decode(
            output[0][len(inputs.input_ids[0]) :], skip_special_tokens=True
        )
    elif generation_args["model_type"] == "groq":
        msg = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            },
        ]
        response = generation_args['client'].chat.completions.create(
            model=generation_args['model'],
            messages=msg
        )
        return response.choices[0].message.content
    else:
        raise ValueError(f"Unsupported model type: {generation_args['model_type']}")


# UTILS TO RUN ON CLUSTER ===========================================

TEMPLATES = {
    "llama2": "<s>[INST] <<SYS>>\n{}\n<</SYS>>\n\n{} [/INST]",
    "llama3": "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
}

def load_model(args: Namespace) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    import huggingface_hub
    
    load_dotenv()
    huggingface_hub.login(token=os.getenv("HF_ACCESS_TOKEN"))
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto" if args.parallel else args.device, 
        torch_dtype=torch.float32 if args.dtype == "f32" else torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    return model, tokenizer  # type: ignore


# UTILS TO USE GROQ API =============================================

def set_client():
    from groq import Groq

    load_dotenv()
    api_key = os.getenv('GROQ_API_KEY')
    client = Groq(api_key=api_key)
    return client


# DEBUGGING AND LOGGING UTILS =======================================

def init_log_files():
    files = ["nlu", "dm", "dst", "dialogue", "full"]
    filenames = {}
    files_text = {}
    for file in files:
        filenames[file] = set_file_name(os.path.join("dialogues", file + ".txt"))
        files_text[file] = ""
    return filenames, files_text

def log_turn(filenames: dict, files_text: dict, turn: int, user_input: str, nlu_outputs: dict, dm_outputs: dict, nlg_output: str, dialogue_state: DialogueState):
    files_text["nlu"] += "\nturn: " + str(turn) + "\n" + json_to_string(nlu_outputs) + "\n"
    files_text["dm"] += "\nturn: " + str(turn) + "\n" + json_to_string(dm_outputs) + "\n"
    files_text["dst"] += "\nturn: " + str(turn) + "\n" + str(dialogue_state) + "\n"
    files_text["dialogue"] += "\nturn: " + str(turn) + "\n" + "user: " + user_input + "\n" + "system: " + nlg_output + "\n"
    files_text["full"] += "\nturn: " + str(turn) + "\n" + "user: " + user_input + "\n" + str(dialogue_state) + "\n" + json_to_string(dm_outputs) + "\n" + "system: " + nlg_output + "\n"
    for file in filenames:
        save_txt(filenames[file], files_text[file])
