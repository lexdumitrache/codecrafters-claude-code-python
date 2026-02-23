import argparse
import os
import sys
import json

from openai import OpenAI

# Read credentials from environment variables (OpenRouter-style).
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

# Advertise BOTH tools: Read + Write.
# The model can decide which ones to call and in what order.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    }
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write content to a file",
            "parameters": {
                "type": "object",
                "required": ["file_path", "content"],
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path of the file to write to",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
            },
        },
    },
]


def tool_read(file_path: str) -> str:
    """
    Implementation of the Read tool.
    Reads a UTF-8 text file and returns its content.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def tool_write(file_path: str, content: str) -> str:
    """
    Implementation of the Write tool.
    Creates the file if it doesn't exist, overwrites it if it does.
    Returns a short confirmation string (tool output).
    """
    # Ensure parent directories exist if the model writes into nested paths.
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    # The tool output can be anything; keep it simple and deterministic.
    return f"Wrote {len(content)} chars to {file_path}"


def run_agent_loop(client: OpenAI, model: str, user_prompt: str) -> str:
    """
    Multi-step agent loop:
    - Maintain conversation history in `messages`.
    - Call the model.
    - If it requests tools, execute them and append tool outputs.
    - Stop when it returns a normal content response (no tool calls).
    """
    messages = [{"role": "user", "content": user_prompt}]

    while True:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
        )

        if not resp.choices:
            raise RuntimeError("no choices in response")

        choice = resp.choices[0]
        message = choice.message

        # Append assistant response to history (JSON-friendly dict).
        assistant_entry = {"role": "assistant", "content": message.content}
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in tool_calls
            ]
        messages.append(assistant_entry)

        # If no tools requested, final answer is in content.
        if not tool_calls:
            return message.content or ""

        # Execute each tool call and append tool result messages.
        for tc in tool_calls:
            fn_name = tc.function.name
            raw_args = tc.function.arguments or "{}"

            try:
                args_obj = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid tool arguments JSON: {e}")

            if fn_name == "Read":
                file_path = args_obj.get("file_path")
                if not file_path:
                    raise RuntimeError("Missing required argument: file_path")

                output = tool_read(file_path)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            elif fn_name == "Write":
                file_path = args_obj.get("file_path")
                content = args_obj.get("content")
                if not file_path:
                    raise RuntimeError("Missing required argument: file_path")
                if content is None:
                    raise RuntimeError("Missing required argument: content")

                output = tool_write(file_path, content)

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )

            else:
                raise RuntimeError(f"Unsupported tool: {fn_name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True, help="Prompt to send to the model")
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    final_text = run_agent_loop(
        client=client,
        model="anthropic/claude-haiku-4.5",
        user_prompt=args.p,
    )

    # Print ONLY final answer to stdout.
    sys.stdout.write(final_text)


if __name__ == "__main__":
    main()