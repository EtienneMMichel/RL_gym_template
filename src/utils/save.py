import os
import yaml
import json

SAVE_DIR = "SAVE"

def save_model(config, agent, saving_path):
    agent.save(saving_path)

def save_model_performance(infos, saving_path):
    with open(f"{saving_path}/infos.json", 'w', encoding='utf-8') as f:
        json.dump(infos, f, ensure_ascii=False, indent=4)


def save(config, agent, infos):
    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)
    saving_path = f"{SAVE_DIR}/{len(os.listdir(SAVE_DIR))}"
    os.mkdir(saving_path)
    with open('config.yaml', 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)

    save_model(config, agent, saving_path)
    save_model_performance(infos, saving_path)