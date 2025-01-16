import json

def convert_json_format(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        item['instruction'] = item.pop('query')
        item['input'] = ''
        item['output'] = item.pop('response')

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

input_file = './dataset/ruozhiba_qaswift.json'
output_file = './dataset/ruozhiba.json'

convert_json_format(input_file, output_file)