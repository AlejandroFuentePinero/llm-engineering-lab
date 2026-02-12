import time
from dotenv import load_dotenv
import os
from openai import OpenAI
import subprocess
from llm_code_performance_benchmark.src.system_info import retrieve_system_info

# --- C++ compilator
compile_command = [
    "clang++",
    "-std=c++17",
    "-Ofast",
    "-mcpu=native",
    "-flto=thin",
    "-fvisibility=hidden",
    "-DNDEBUG",
    "main.cpp",
    "-o",
    "main",
]

# --- Prompts
system_prompt = """
    Your task is to convert Python code into high performance C++ code.
    Respond only with C++ code. Do not provide any explanation other than occasional comments.
    The C++ response needs to produce an identical output in the fastest possible time.
    """


def user_prompt_for(python, system_info, model):
    return f"""
    Port this Python code to C++ with the fastest possible implementation that produces identical output in the least time.
    The system information is:
    {system_info}
    Your response will be written to a file called {model}_main.cpp and then compiled and executed; the compilation command is:
    {compile_command}
    Respond only with C++ code.
    Python code to port:

    ```python
    {python}
    ```
    """


# --- Helper functions


def messages_for(python, system_info, model):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt_for(python, system_info, model)},
    ]


def write_output(cpp: str, model: str):
    # file used for compilation
    with open("main.cpp", "w", encoding="utf-8") as f:
        f.write(cpp)

    # archive per model (safe filename)
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model)
    with open(f"{safe}_main.cpp", "w", encoding="utf-8") as f:
        f.write(cpp)


def port(client, model, python, system_info):
    kwargs = {
        "model": model,
        "messages": messages_for(python, system_info, model),
    }
    if "gpt" in model:
        kwargs["reasoning_effort"] = "high"

    response = client.chat.completions.create(**kwargs)
    reply = response.choices[0].message.content or ""
    reply = reply.replace("```cpp", "").replace("```", "").strip()
    write_output(reply, model)


def run_python(code):
    ns = {"__builtins__": __builtins__}
    exec(code, ns)
    return ns["result"], ns["execution_time"]


def compile_and_run(model: str):
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in model)
    cpp_file = f"{safe}_main.cpp"
    bin_file = f"{safe}_main"

    cmd = compile_command.copy()
    cmd[cmd.index("main.cpp")] = cpp_file
    cmd[cmd.index("main")] = bin_file

    try:
        subprocess.run(cmd, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        print("COMPILER STDOUT:\n", e.stdout)
        print("COMPILER STDERR:\n", e.stderr)
        raise

    run = subprocess.run([f"./{bin_file}"], check=True, text=True, capture_output=True)
    print(run.stdout)
    return run.stdout


# --- Code example
pi = """
import time

def calculate(iterations, param1, param2):
    result = 1.0
    for i in range(1, iterations+1):
        j = i * param1 - param2
        result -= (1/j)
        j = i * param1 + param2
        result += (1/j)
    return result

start_time = time.perf_counter()
result = calculate(200_000_000, 4, 1) * 4
end_time = time.perf_counter()
print("Performance:")
print("------------")
print(f"Result: {result:.12f}")
print(f"Execution Time: {(end_time - start_time):.6f} seconds")
execution_time = end_time - start_time
"""


def python_to_cpp_performance(
    claude_model: str = "claude-sonnet-4-5-20250929",
    openai_model: str = "gpt-5",
    python: str = pi,
    check_API_key: bool = False,
):
    # --- Load keys
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    if check_API_key:
        if openai_api_key:
            print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
        else:
            print("OpenAI API Key not set")

        if anthropic_api_key:
            print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
        else:
            print("Anthropic API Key not set")

    # --- Define client
    openai = OpenAI()

    anthropic_url = "https://api.anthropic.com/v1/"
    anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)

    # --- Retrieve system info
    system_info = retrieve_system_info()

    # --- Run the code in python and return performance (runtime)
    py_res, py_time = run_python(python)

    # --- Call OpenAI to generate optimise C++ code
    port(openai, openai_model, python, system_info)
    openai_res = compile_and_run(openai_model)
    openai_code_lag = openai_res.split("Result: ")[1]
    openai_code_res = openai_code_lag.split("\nExecution")[0]
    openai_lag = openai_res.split("Time: ")[1]
    openai_time = openai_lag.split(" seconds")[0]
    openai_performance = py_time / float(openai_time)

    # --- Call Anthropic to generate optimise C++ code
    port(anthropic, claude_model, python, system_info)
    claude_res = compile_and_run(claude_model)
    claude_code_lag = claude_res.split("Result: ")[1]
    claude_code_res = claude_code_lag.split("\nExecution")[0]
    claude_lag = claude_res.split("Time: ")[1]
    claude_time = claude_lag.split(" seconds")[0]
    claude_performance = py_time / float(claude_time)

    out = {
        "Python result": py_res,
        "Python runtime": py_time,
        "OpenAI result": openai_code_res,
        "OpenAI runtime": openai_time,
        "OpenAI performance": openai_performance,
        "Claude result": claude_code_res,
        "Claude runtime": claude_time,
        "Claude performance": claude_performance,
    }

    print(
        f"AI Performance:\n"
        f"---------------\n"
        f"Claude (model: {claude_model}): {claude_performance}X speedup\n"
        f"OpenAI (model: {openai_model}): {openai_performance}X speedup"
    )
    return out
