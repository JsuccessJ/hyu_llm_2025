import json
import random

def load_perfume_data(json_path):
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data.get('perfumes_data', [])

def filter_perfume_keys(perfumes_data):
    keys_to_extract = [
        'name', 'target', 'brand_name', 'accords',
        'short_description', 'detailed_description',
        'seasonal', 'time_of_day', 'pros', 'cons'
    ]
    filtered_perfumes = [
        {key: perfume.get(key, None) for key in keys_to_extract}
        for perfume in perfumes_data
    ]
    return filtered_perfumes

def generate_instruction_templates(perfumes_data):
    templates_by_type = {
        "strongest_accord": [],
        "description_summary": [],
        "best_season": [],
        "time_of_day": [],
        "description_relationship": []
    }
    failed_items = []

    for perfume in perfumes_data:
        name = perfume.get('name', '')
        brand = perfume.get('brand_name', '')
        missing_template_names = []

        # Template 1: Strongest accord
        if 'accords' in perfume and perfume['accords']:
            try:
                accords = perfume['accords']
                accord_input = ', '.join([f"{a['accord']}: {a['strength']}" for a in accords])
                strongest = max(accords, key=lambda x: float(x['strength'].replace('%', '')))
                templates_by_type['strongest_accord'].append({
                    'instruction': "Answer which of the given scents is the strongest.",
                    'input': accord_input,
                    'output': strongest['accord']
                })
            except Exception:
                missing_template_names.append("strongest_accord")
        else:
            missing_template_names.append("strongest_accord")

        # Template 2: Description summary
        if perfume.get('detailed_description') and perfume.get('short_description'):
            templates_by_type['description_summary'].append({
                'instruction': "Summarize the description of the entered perfume.",
                'input': perfume['detailed_description'],
                'output': perfume['short_description']
            })
        else:
            missing_template_names.append("description_summary")

        # Template 3: Most appropriate season
        seasonal = perfume.get('seasonal')
        if seasonal:
            valid_seasons = [(k, v) for k, v in seasonal.items() if v is not None]
            if valid_seasons:
                try:
                    best_season = max(valid_seasons, key=lambda x: float(x[1].replace('%', '')))[0]
                    templates_by_type['best_season'].append({
                        'instruction': "Given a numerical representation of the seasons for which a perfume is appropriate, answer which season is the most appropriate.",
                        'input': ', '.join([f"{k}: {v}" for k, v in seasonal.items()]),
                        'output': best_season
                    })
                except Exception:
                    missing_template_names.append("best_season")
            else:
                missing_template_names.append("best_season")
        else:
            missing_template_names.append("best_season")

        # Template 4: Most appropriate time of day
        tod = perfume.get('time_of_day')
        if tod:
            valid_times = [(k, v) for k, v in tod.items() if v is not None]
            if valid_times:
                try:
                    best_time = max(valid_times, key=lambda x: float(x[1].replace('%', '')))[0]
                    templates_by_type['time_of_day'].append({
                        'instruction': "Given a numerical representation of the time of day when a perfume is appropriate to wear, answer the appropriate time of day.",
                        'input': ', '.join([f"{k}: {v}" for k, v in tod.items()]),
                        'output': best_time
                    })
                except Exception:
                    missing_template_names.append("time_of_day")
            else:
                missing_template_names.append("time_of_day")
        else:
            missing_template_names.append("time_of_day")

        # Template 5: Description relationship
        if perfume.get('detailed_description') and perfume.get('short_description'):
            templates_by_type['description_relationship'].append({
                'instruction': "Determine the relationship between the following given sentence A and B.",
                'input': f"Sentence A: {perfume['detailed_description']}. Sentence B: {perfume['short_description']}.",
                'output': "Summarization"
            })
        else:
            missing_template_names.append("description_relationship")

        if missing_template_names:
            failed_items.append({
                'name': name,
                'brand': brand,
                'missing_templates': missing_template_names
            })

    return templates_by_type, failed_items

def split_templates_for_test(templates_by_type, split_ratio=0.9):
    train_data = []
    test_templates_by_type = {}

    for key, templates in templates_by_type.items():
        if key == "description_relationship":
            train_data.extend(templates)
        else:
            random.shuffle(templates)
            split_idx = int(len(templates) * split_ratio)
            train_data.extend(templates[:split_idx])
            test_templates_by_type[key] = templates[split_idx:]

    return train_data, test_templates_by_type

def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    json_path = input("Enter the path to the perfume JSON file: ").strip()
    perfumes_data = load_perfume_data(json_path)
    filtered_perfumes = filter_perfume_keys(perfumes_data)
    templates_by_type, failed_items = generate_instruction_templates(filtered_perfumes)

    train_data, test_templates_by_type = split_templates_for_test(templates_by_type)

    # 저장
    save_json(train_data, 'train_templates.json')
    print(f"Train templates saved: {len(train_data)} → train_templates.json")

    for template_name, test_data in test_templates_by_type.items():
        filename = f"test_{template_name}.json"
        save_json(test_data, filename)
        print(f"Test templates ({template_name}): {len(test_data)} → {filename}")

    save_json(failed_items, 'failed_items.json')
    print(f"Failed items saved → failed_items.json")
