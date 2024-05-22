import yaml
import json
from datetime import datetime
import os

now = datetime.now()
formatted_date_time = now.strftime('%Y%m%d-%H%M%S')


def save(file_name: str, data: dict):
    dir = os.path.dirname(file_name)

    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(file_name, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
        f.close()
