import argparse
import os
import sys

from openai import OpenAI

API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True)
    args = p.parse_args()

    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    chat = client.chat.completions.create(
        model="anthropic/claude-haiku-4.5",
        messages=[{"role": "user", "content": args.p}],
        tools=[{
            "type": "function",
            "function": {
                "name": "Read",
                "description": "Read and return the contents of a file",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "The path to the file to read"
                            }
                        },
                    "required": ["file_path"]
                    }
                }
            }
        ],
    )

    if not chat.choices or len(chat.choices) == 0:
        raise RuntimeError("no choices in response")

    # You can use print statements as follows for debugging, they'll be visible when running tests.
    print("Logs from your program will appear here!", file=sys.stderr)

    message = chat.choices[0].message

    # Tool-call path
    if getattr(message, "tool_calls", None):
        tool_call = message.tool_calls[0]
        fn_name = tool_call.function.name

        if fn_name != "Read":
            raise RuntimeError(f"Unsupported tool call: {fn_name}")

        try:
            args_obj = json.loads(tool_call.function.arguments or "{}")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Invalid tool arguments JSON: {e}")

        file_path = args_obj.get("file_path")
        if not file_path:
            raise RuntimeError("Missing required argument: file_path")

        with open(file_path, "r", encoding="utf-8") as f:
            sys.stdout.write(f.read())
        return

    # Normal content path
    if message.content:
        sys.stdout.write(message.content)
        return

    # If neither tool_calls nor content exists
    raise RuntimeError("No tool_calls or content in response")


if __name__ == "__main__":
    main()
