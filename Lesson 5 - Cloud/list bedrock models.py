import boto3
from botocore.exceptions import ClientError, BotoCoreError

# ---
# Script to list all Amazon Bedrock foundation models available in a given AWS region
# using a specific AWS profile. Prints out model details and access status.
# ---

# Create a boto3 session using the 'robbarto' AWS profile and the 'us-east-1' region.
# This ensures the script runs with your credentials and correct permissions.
session = boto3.Session(profile_name="robbarto", region_name="us-east-1")

# Create a client for the Bedrock service (not bedrock-runtime, which is for inference).
bedrock = session.client("bedrock")

try:
    # Call the Bedrock API to list all foundation models you can see in the region.
    response = bedrock.list_foundation_models()
    # Extract the list of model summaries from the response.
    model_summaries = response.get("modelSummaries", [])

    if model_summaries:
        # Print the number of models found and their details in a readable format.
        print(f"✅ {len(model_summaries)} Bedrock models found in us-east-1:")
        for model in model_summaries:
            model_id = model.get("modelId", "N/A")               # Unique model identifier
            provider = model.get("providerName", "N/A")          # Model provider name (e.g., Amazon, Anthropic)
            name = model.get("modelName", "N/A")                 # Human-readable model name
            access = model.get("modelAccess", "UNKNOWN")         # Access status (ENABLED, NOT_ENABLED, REQUESTED, or UNKNOWN)
            print(f"- {model_id} | {name} | {provider} | Access: {access}")
    else:
        # Handle the case where no models are returned (e.g., permissions issue or empty region)
        print("⚠️ No Bedrock models returned in us-east-1.")

except (ClientError, BotoCoreError) as e:
    # Catch and display AWS/boto3 errors (e.g., credentials, permissions, API errors)
    print(f"❌ Error listing Bedrock models: {e}")

except (ClientError, BotoCoreError) as e:
    print(f"❌ Error listing Bedrock models: {e}")
