import yaml
import json

name = 'zurich_city_11_c'
# 读取 YAML 文件并将其转换为 Python 字典对象
with open("I:/DSEC/" + name +"/calibration/cam_to_cam.yaml", 'r') as stream:
    data = yaml.safe_load(stream)

# 将 Python 字典对象转换为 JSON 字符串
json_data = json.dumps(data)

with open('I:/DSEC/' + name + '/cam_to_cam.json', 'w') as f:
    f.write(json_data)
