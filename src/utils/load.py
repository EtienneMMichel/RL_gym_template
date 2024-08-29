import os
import yaml
import json
from .save import SAVE_DIR




def load_model(agent, folder_name):
    loading_path = f"{SAVE_DIR}/{folder_name}"
    agent.from_save(loading_path)