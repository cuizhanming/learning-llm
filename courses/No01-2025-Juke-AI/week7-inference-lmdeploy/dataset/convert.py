import json

def convert_json_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        item['question'] = item.pop('query')
        item['answer'] = item.pop('response')

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

input_file = 'ruozhiba_qaswift.json'
output_file = 'ruozhiba.jsonl'

convert_json_format(input_file, output_file)