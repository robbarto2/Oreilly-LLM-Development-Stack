import boto3
import json

# Create a session using your profile and region
session = boto3.Session(profile_name="robbarto", region_name="us-east-1")
bedrock_runtime = session.client("bedrock-runtime")

# Claude 3.5 Sonnet Model ID (latest known format as of June 2025)
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Create the Claude-style prompt
prompt = {
    "messages": [
        {
            "role": "user",
            "content": "explain where the james webb space telescope is located."
        }
    ],
    "max_tokens": 512,
    "temperature": 0.7,
    "anthropic_version": "bedrock-2023-05-31"
}

try:
    invoke_kwargs = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": json.dumps(prompt)
    }
    response = bedrock_runtime.invoke_model(**invoke_kwargs)

    # Parse and print the response
    result = json.loads(response["body"].read())
    print("✅ Response from Claude 3.5 Sonnet:\n")
    print(result["content"][0]["text"])

except Exception as e:
    print("❌ Error invoking model:", e)
