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
    # Build the chat messages list with a system prompt
    messages = [
        {"role": "system", "content": "You are a smart assistant in the style of Carl Sagan."},
        {"role": role, "content": f"{user}: {content}"}
    ]
    response = ollama.chat(model=model, messages=messages)
    return response["message"]["content"]


def terminal_chat(model="gemma3:4b"):
    """
    Start an interactive chat session with the Ollama model from the terminal.
    Type 'exit' to quit.
    """
    messages = [
        {"role": "system", "content": "You are a smart assistant in the style of Carl Sagan."}
    ]
    print("Type 'exit' to quit.")
    import datetime
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

    # Save chat history to a file when session ends
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"ollama_chat_{timestamp}.txt"
    with open(filename, 'w', encoding='utf-8') as f:
        for msg in messages:
            f.write(f"{msg['role'].capitalize()}: {msg['content']}\n\n")
    print(f"Chat history saved to {filename}")

if __name__ == "__main__":
    terminal_chat()