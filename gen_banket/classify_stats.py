import os
import json
from collections import Counter

input_folder = "/home/team_cv/tdkien/20_Vuon_Minh/annotations/classify_thuc_chien_key"

# Counters for each field
mon_hoc_counter = Counter()
do_kho_counter = Counter()
cap_hoc_counter = Counter()

# Get list of json files
json_files = [f for f in os.listdir(input_folder) if f.endswith('.json')]

total_files = len(json_files)

for file_name in json_files:
    file_path = os.path.join(input_folder, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        mon_hoc_counter[data['mon_hoc']] += 1
        do_kho_counter[data['do_kho']] += 1
        cap_hoc_counter[data['cap_hoc']] += 1

print(f"Total samples: {total_files}")
print()

print("Mon hoc (Subject) statistics:")
for key, count in mon_hoc_counter.most_common():
    proportion = count / total_files * 100
    print(f"  {key}: {count} ({proportion:.2f}%)")
print()

print("Do kho (Difficulty) statistics:")
for key, count in do_kho_counter.most_common():
    proportion = count / total_files * 100
    print(f"  {key}: {count} ({proportion:.2f}%)")
print()

print("Cap hoc (Level) statistics:")
for key, count in cap_hoc_counter.most_common():
    proportion = count / total_files * 100
    print(f"  {key}: {count} ({proportion:.2f}%)")
