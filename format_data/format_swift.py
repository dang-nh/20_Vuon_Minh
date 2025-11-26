import os
import json

input_folder = "/home/team_cv/tdkien/20_Vuon_Minh/annotations/labels_train_thuc_chien_key"
jsonl_output1 = "/home/team_cv/tdkien/20_Vuon_Minh/annotations/labels_train_thuc_chien_key_1.jsonl"
jsonl_output2 = "/home/team_cv/tdkien/20_Vuon_Minh/annotations/labels_train_thuc_chien_key_2.jsonl"
jsonl_output3 = "/home/team_cv/tdkien/20_Vuon_Minh/annotations/labels_train_thuc_chien_key_3.jsonl"

for filename in os.listdir(input_folder):
    # Group 1
    if filename.endswith("1.json"):
        data = json.load(open(os.path.join(input_folder, filename), 'r', encoding='utf-8'))
        system_prompt = data["system_prompt"]["prompt"]
        conversation = data["hoc_tap_tuong_tac"]["conversations"]
        if len(conversation) % 2 != 0:
            conversation = conversation[:-1] # Ensure even number of messages
        messages = [{"role": "system", "content": system_prompt}]
        for i, msg in enumerate(conversation):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append({"role": role, "content": msg})
        hoc_tap_tuong_tac = {"messages": messages}
        
        hoc_tap_ca_nhan_hoa = {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data["hoc_tap_ca_nhan_hoa"]["question"]},
            {"role": "assistant", "content": data["hoc_tap_ca_nhan_hoa"]["answer"]}
        ]}
        
        goi_mo_y_tuong = {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data["goi_mo_y_tuong"]["question"]},
            {"role": "assistant", "content": data["goi_mo_y_tuong"]["answer"]}
        ]}

        ho_tro_tam_ly_cam_xuc = {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data["ho_tro_tam_ly_cam_xuc"]["question"]},
            {"role": "assistant", "content": data["ho_tro_tam_ly_cam_xuc"]["answer"]}
        ]}

        tu_van_an_toan = {"messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": data["tu_van_an_toan"]["question"]},
            {"role": "assistant", "content": data["tu_van_an_toan"]["answer"]}
        ]}

        with open(jsonl_output1, 'a', encoding='utf-8') as f:
            for item in [hoc_tap_tuong_tac, hoc_tap_ca_nhan_hoa, goi_mo_y_tuong, ho_tro_tam_ly_cam_xuc, tu_van_an_toan]:
                json_line = json.dumps(item, ensure_ascii=False)
                f.write(json_line + '\n')

    # Group 2
    elif filename.endswith("2.json"):
        data = json.load(open(os.path.join(input_folder, filename), 'r', encoding='utf-8'))
        system_prompt = data["system_prompt"]["prompt"]
        with open(jsonl_output2, 'a', encoding='utf-8') as f:
            system_prompt = data["system_prompt"]["prompt"]
            for key in data:
                if key != "system_prompt":
                    item = {"messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": data[key]["question"]},
                        {"role": "assistant", "content": data[key]["answer"]}
                    ]}
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + '\n')
    
    # Group 3
    elif filename.endswith("3.json"):
        data = json.load(open(os.path.join(input_folder, filename), 'r', encoding='utf-8'))
        system_prompt = data["system_prompt"]["prompt"]
        with open(jsonl_output3, 'a', encoding='utf-8') as f:
            system_prompt = data["system_prompt"]["prompt"]
            for key in data:
                if key != "system_prompt":
                    item = {"messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": data[key]["question"]},
                        {"role": "assistant", "content": data[key]["answer"]}
                    ]}
                    json_line = json.dumps(item, ensure_ascii=False)
                    f.write(json_line + '\n')

