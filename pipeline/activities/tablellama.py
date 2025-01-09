import os
import sys
import json
import re
import io
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# Using pandas to read some structured data
import pandas as pd
from io import StringIO

sys.path.append('./..')


def read_dicts_from_file(file_name):
    """
    Read a file with each line containing a JSON string representing a dictionary,
    and return a list of dictionaries.

    :param file_name: Name of the file to read from.
    :return: List of dictionaries.
    """
    dict_list = []
    with open(file_name, 'r') as file:
        for line in file:
            # Convert the JSON string back to a dictionary.
            dictionary = json.loads(line.rstrip('\n'))
            dict_list.append(dictionary)
    return dict_list

def get_model():

    model_name = "osunlp/TableLlama"
    config = AutoConfig.from_pretrained(
        model_name,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(32001)

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # cache_dir=args.cache_dir,
        # model_max_length=args.context_size if args.context_size > orig_ctx_len else orig_ctx_len,
        padding_side="left",
        use_fast=False,
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    return model, tokenizer


def check_for_python_code(response, df):
    python_output = io.StringIO()
    sys.stdout = python_output
    code_block = re.search(r'```python(.*?)```', response, re.DOTALL) 
    local_vars = {
        'df' : df
    }
    if code_block: 
        python_code = code_block.group(1).strip() 
        try:
            exec(python_code, {}, local_vars)
        except Exception as e:
            sys.__stdout__
            print(python_code)
            print(f"An error occurred: {e}")
            return None
        sys.stdout = sys.__stdout__  # Reset standard output
        return python_output.getvalue()
    return None


def run_benchmarks():
    model, tokenizer = get_model()
    example_prompt_template = """Given access to several pandas dataframes, write the Python code to answer the user's question.

    /*
    "{var_name}.head(5).to_string(index=False)" as follows:
    {df_info}
    */

    Question: {user_question}
    Constraints: {constraints}
    """

    benchmarks = read_dicts_from_file('../../examples/DA-Agent/data/da-dev-questions.jsonl')
    table_path = '../../examples/DA-Agent/data/da-dev-tables'
    results = []

    i = 0

    for q in benchmarks:
        i += 1
        print(f"benchmark {q['id']}")
        user_question = q['question']
        concepts = q['concepts']
        file_path = q['file_name']
        constraints = q['constraints']
        format_ = q['format']

        file_path = os.path.join(table_path, file_path)
        df = pd.read_csv(file_path)

        prompt = example_prompt_template.format(
            var_name="df",
            df_info=df.head(5).to_string(index=False),
            user_question=user_question,
            constraints=constraints
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.6,
            top_p=.9,
        )
        response = tokenizer.decode(output[0], skip_special_tokens=False, clean_up_tokenization_spaces=False)

        response = response.split(prompt)[1].strip()
        if response.endswith('</s>'):
            response = response[:len(response) - 4]

#         python_code_output = check_for_python_code(response, df)
#         response = f'''
# {response}
# Output:
# {python_code_output}
#         '''

        iteration_result = {
            'id': q['id'],
            'input_text': prompt,
            'concepts': concepts,
            'file_path': file_path,
            'response': response,
            'format': format_
        }
        results.append(iteration_result)

        if i % 10 == 0:
            with open('results_tablellama.json', 'w') as outfile:
                json.dump(results, outfile, indent=4)


    with open('results_tablellama.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == '__main__':
    run_benchmarks()
