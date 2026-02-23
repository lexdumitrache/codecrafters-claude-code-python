import argparse
import os
import sys
import json
import subprocess

from openai import OpenAI

# Read credentials from environment variables (OpenRouter-style).
API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = os.getenv("OPENROUTER_BASE_URL", default="https://openrouter.ai/api/v1")

# Advertise tools: Read + Write + Bash.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read and return the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "The path to the file to read"}
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
                    "file_path": {"type": "string", "description": "The path of the file to write to"},
                    "content": {"type": "string", "description": "The content to write to the file"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Execute a shell command",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"},
                },
            },
        },
    },
]


def tool_read(file_path: str) -> str:
    """Read a UTF-8 text file and return its contents."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def tool_write(file_path: str, content: str) -> str:
    """Create/overwrite a UTF-8 text file with provided content."""
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)

    return f"Wrote {len(content)} chars to {file_path}"


def tool_bash(command: str) -> str:
    """
    Execute a shell command in the CURRENT working directory (important for tests).
    Capture stdout and stderr and return them to the model.
    """
    try:
        completed = subprocess.run(
            command,
            shell=True,              # command is a string, run via shell
            capture_output=True,     # capture stdout/stderr
            text=True,               # decode to string
            cwd=os.getcwd(),         # ensure we run where the program is executed
        )
    except Exception as e:
        return f"ERROR: failed to run command: {e}"

    # Return both stdout and stderr so the model can reason about failures.
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    if completed.returncode != 0:
        return f"ERROR (code {completed.returncode})\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"

    # Successful commands often return empty output (e.g., rm file).
    # Still return both streams for completeness.
    combined = ""
    if stdout:
        combined += stdout
    if stderr:
        # Some commands output warnings on stderr even when successful.
        combined += ("" if combined.endswith("\n") or combined == "" else "\n") + stderr

    return combined


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
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
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

            elif fn_name == "Write":
                file_path = args_obj.get("file_path")
                content = args_obj.get("content")
                if not file_path:
                    raise RuntimeError("Missing required argument: file_path")
                if content is None:
                    raise RuntimeError("Missing required argument: content")
                output = tool_write(file_path, content)

            elif fn_name == "Bash":
                command = args_obj.get("command")
                if not command:
                    raise RuntimeError("Missing required argument: command")
                output = tool_bash(command)

            else:
                raise RuntimeError(f"Unsupported tool: {fn_name}")

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": output,
                }
            )


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

    # Print ONLY final answer to stdout (tests usually compare stdout exactly).
    sys.stdout.write(final_text)


if __name__ == "__main__":
    main()