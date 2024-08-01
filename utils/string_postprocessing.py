# Import libraries
import json


# Postprocessing of JSON
def jsonify(json_string: str):
    start = json_string.find('{')
    end = json_string.rfind('}')
    json_string = json_string[start:end+1]
    json_obj = json.loads(json_string)
    return json_obj