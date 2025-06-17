import ollama
import json

def send_message_to_ollama(role, user, content, model="gemma3:4b"):
    """
    Send a message to the Ollama model using the ollama Python library.
    Args:
        role (str): Role of the sender (e.g., 'user', 'assistant')
        user (str): User identifier or name
        content (str): Message content
        model (str): Model name to use (default: 'gemma3:4b')
    Returns:
        str: Ollama model response
    """
    # Format the prompt to include role and user
    prompt = f"[{role}] {user}: {content}"
    response = ollama.generate(model=model, prompt=prompt)
    return response.get("response", "")


if __name__ == "__main__":
    # Example usage
    role = "user"
    user = "alice"
    content = "Where is the James Web Space Telescope?"

    try:
        response = send_message_to_ollama(role, user, content)
        print("Ollama response:")
        print(response)
    except Exception as e:
        print(f"Error communicating with Ollama: {e}")