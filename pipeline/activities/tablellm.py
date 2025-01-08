import os
import sys
import json
import re
import io
#import sklearn

from vllm import LLM, SamplingParams

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



def check_for_python_code(response, df):
    response = "\n".join([line for line in response.split("\n") if "pd.read_csv" not in line])

    python_output = io.StringIO()
    sys.stdout = python_output
    local_vars = {
        'df' : df
    }
    try:
        exec(response, {}, local_vars)
    except Exception as e:
        print(f"An error occurred: {e}")
    sys.stdout = sys.__stdout__  # Reset standard output
    return python_output.getvalue()



def run_benchmarks():
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

    # load model
    llm = LLM(
        dtype="half", 
        model="RUCKBReasoning/TableLLM-13b", 
        tensor_parallel_size=1,
        max_model_len=2048
        )

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

        prompt = f"[INST]{prompt}[/INST]"


        # get LLM response
        sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=512)
        responses = llm.generate([prompt], sampling_params=sampling_params)
        response = responses[0].outputs[0].text.lstrip(' ')


        python_code_output = check_for_python_code(response, df)
        response = f'''
{response}
Output:
{python_code_output}
        '''

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
            with open('results_tablellm.json', 'w') as outfile:
                json.dump(results, outfile, indent=4)


    with open('results_tablellm.json', 'w') as outfile:
        json.dump(results, outfile, indent=4)

if __name__ == '__main__':
    run_benchmarks()
