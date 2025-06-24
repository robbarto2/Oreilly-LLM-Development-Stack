import requests
import json

url = "http://localhost:11434/api/generate"

data = {
        "model": "gemma3:4b",
        "prompt": "tell me a short story about Ireland and make it funny"
}

response = requests.post(url, json=data, stream=True)

#check the response
if response.status_code == 200:
    print("Generated Text:", end=" ", flush=True)
    #iterate over the response
    for line in response.iter_lines():
        if line:
            #decode the line
            decoded_line = line.decode("utf-8")
            result = json.loads(decoded_line)
            #get the text from the response
            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)
else:
    print("Error:", response.status_code, response.text)
