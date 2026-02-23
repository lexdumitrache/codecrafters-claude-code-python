[![progress-banner](https://backend.codecrafters.io/progress/claude-code/d3d3965a-90d4-4ef3-b6fd-0e770d78abbe)](https://app.codecrafters.io/users/codecrafters-bot?r=2qF)

# Claude Code — LLM-Powered Coding Assistant (Python)

This repository contains **my implementation of a simplified “Claude Code”**, an AI-powered coding assistant built from scratch in Python as part of the  
[Build Your Own Claude Code challenge by CodeCrafters](https://codecrafters.io/challenges/claude-code).

The project focuses on designing a **tool-calling LLM agent** capable of reasoning, executing actions, and completing multi-step tasks through an agent loop.

---

## 🧠 What this project does

The assistant communicates with a Large Language Model (LLM) using an **OpenAI-compatible API** (via OpenRouter) and supports multiple tools that allow it to interact with the local environment.

Implemented tools include:

- **Read** — read the contents of files  
- **Write** — create or overwrite files  
- **Bash** — execute shell commands  

The model decides when and how to use these tools, while the program executes them deterministically and feeds results back into the conversation.

---

## 🔁 Agent Loop

A key part of the implementation is the **agent loop**, which enables multi-step reasoning:

1. The user provides a prompt
2. The model responds, optionally requesting one or more tool calls
3. Requested tools are executed by the program
4. Tool results are appended to the conversation
5. The loop repeats until the model produces a final response

This allows the assistant to solve tasks such as:

- Reading instructions from a file and acting on them
- Creating or modifying files based on context
- Cleaning up or manipulating project files via shell commands

---

## 🏗️ Architecture

- **Language:** Python  
- **API:** OpenAI-compatible (OpenRouter)  
- **Core concepts implemented:**
  - Tool calling schema
  - Persistent conversation state
  - Multi-step agent loop
  - Deterministic execution of side effects
  - Clean separation between reasoning and execution

The main entry point is in `app/main.py`. All logic — prompt handling, tool execution, and the agent loop — is implemented there.

---

## 🚀 How to run

### Prerequisites
- Python 3.10+
- An OpenRouter API key

Set the required environment variables:

```bash
export OPENROUTER_API_KEY="your_api_key"
export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
```
Run the assistant:

```bash
./your_program.sh -p "Your prompt here"
```

Example:

```bash
./your_program.sh -p "Delete the old readme file. Always respond with 'Deleted README_old.md'"
```
