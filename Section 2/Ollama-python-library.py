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
    # Build the chat messages list
    messages = [
        {"role": role, "content": f"{user}: {content}"}
    ]
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def terminal_chat(model="gemma3:4b"):
    """
    Start an interactive chat session with the Ollama model from the terminal.
    Type 'exit' to quit.
    """
    messages = []
    print("Type 'exit' to quit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        messages.append({"role": "user", "content": user_input})
        try:
            response = ollama.chat(model=model, messages=messages)
            reply = response["message"]["content"]
            print(f"Ollama: {reply}")
            messages.append({"role": "assistant", "content": reply})
        except Exception as e:
            print(f"Error communicating with Ollama: {e}")

if __name__ == "__main__":
    terminal_chat()