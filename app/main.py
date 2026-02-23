import argparse
import os
import sys
import json

from openai import OpenAI

# Read credentials from environment variables (OpenRouter-style).
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

# Define the tool schema ONCE and reuse it in every loop iteration.
# This is the "contract" you give the model: what tools exist + how to call them.
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
    }
]


def tool_read(file_path: str) -> str:
    """
    Implementation of the Read tool.
    The model may ask us to call Read({file_path: ...}).
    Our job is to actually read the file and return its contents as a string.
    """
    # Keep output deterministic and raw (tests typically expect exact contents).
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def run_agent_loop(client: OpenAI, model: str, user_prompt: str) -> str:
    """
    Agent loop:
    - Keep a persistent `messages` conversation history.
    - Repeatedly call the model with the current messages + tool schemas.
    - If the model requests tool calls, execute them and append tool results.
    - Stop when the model returns a normal content response (no tool_calls).
    """

    # 1) Initialize the conversation with the user's prompt.
    messages = [{"role": "user", "content": user_prompt}]

    # 2) Loop until we get a final response with no tool calls.
    while True:
        # Send the current conversation to the model.
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,  # Advertise tools to the model
            # tool_choice="auto"  # default; optional
        )

        # Basic safety: ensure we have at least one choice.
        if not resp.choices:
            raise RuntimeError("no choices in response")

        choice = resp.choices[0]
        message = choice.message  # assistant message object (SDK type)

        # 3) Append the assistant message to conversation history.
        # We store it as a plain dict to keep `messages` JSON-serializable and stable.
        assistant_entry = {
            "role": "assistant",
            "content": message.content,  # often None if tool_calls are present
        }

        # If the assistant included tool calls, capture them in a JSON-friendly way.
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            assistant_entry["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,  # JSON string
                    },
                }
                for tc in tool_calls
            ]

        messages.append(assistant_entry)

        # 4) If there are NO tool calls, we are done: return the assistant's text.
        if not tool_calls:
            # Some SDKs could return None content; treat as empty string.
            return message.content or ""

        # 5) Otherwise, execute each tool call in order and append tool results.
        for tc in tool_calls:
            fn_name = tc.function.name
            raw_args = tc.function.arguments or "{}"

            # Tool arguments come as a JSON *string* → must parse.
            try:
                args_obj = json.loads(raw_args)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"Invalid tool arguments JSON: {e}")

            # Dispatch to the correct tool implementation.
            if fn_name == "Read":
                file_path = args_obj.get("file_path")
                if not file_path:
                    raise RuntimeError("Missing required argument: file_path")

                output = tool_read(file_path)

                # Append tool output message.
                # IMPORTANT: role MUST be "tool" and tool_call_id MUST match the call id.
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": output,
                    }
                )
            else:
                # If the model calls an unsupported tool, fail clearly.
                raise RuntimeError(f"Unsupported tool: {fn_name}")


def main():
    # Parse CLI args.
    p = argparse.ArgumentParser()
    p.add_argument("-p", required=True, help="Prompt to send to the model")
    args = p.parse_args()

    # Ensure key is present.
    if not API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY is not set")

    # Create OpenAI client pointing at OpenRouter.
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Run the multi-step agent loop.
    final_text = run_agent_loop(
        client=client,
        model="anthropic/claude-haiku-4.5",
        user_prompt=args.p,
    )

    # Print ONLY the final answer to stdout (tests usually compare stdout exactly).
    sys.stdout.write(final_text)


if __name__ == "__main__":
    main()