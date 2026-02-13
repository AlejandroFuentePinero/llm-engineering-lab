import time
from dotenv import load_dotenv
import os
from openai import OpenAI
import subprocess
from llm_code_performance_benchmark.src.system_info import retrieve_system_info
import io
import sys

# Optional UI dependency (only needed when ui_launch=True)
try:
    import gradio as gr
except Exception:
    gr = None


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
    return reply  # useful for UI


def run_python(code: str, *, return_stdout: bool = False):
    ns = {"__builtins__": __builtins__}

    buffer = io.StringIO()
    old_stdout = sys.stdout
    sys.stdout = buffer

    try:
        exec(code, ns)
        stdout = buffer.getvalue()

        # Pull structured outputs (function contract)
        try:
            result = ns["result"]
            execution_time = ns["execution_time"]
        except KeyError as e:
            available = ", ".join(
                sorted(k for k in ns.keys() if not k.startswith("__"))
            )
            raise KeyError(
                f"Executed code did not set required variable {e}. "
                f"Expected 'result' and 'execution_time'. "
                f"Available names: {available}"
            ) from None

        if return_stdout:
            return result, execution_time, stdout
        return result, execution_time

    except Exception as e:
        stdout = buffer.getvalue()
        raise RuntimeError(
            f"Error while executing code: {e}\nCaptured stdout:\n{stdout}"
        ) from e

    finally:
        sys.stdout = old_stdout


# --- Minimal attribution support (LLM vs infra/tooling)


class LLMCompileError(RuntimeError):
    """Compilation failed for model-produced C++ (likely invalid C++ output)."""


class LLMRuntimeError(RuntimeError):
    """Binary ran but failed (crash / non-zero exit)."""


def _is_infra_compile_error(stderr: str) -> bool:
    s = (stderr or "").lower()
    infra_markers = [
        "command not found",
        "no such file or directory",
        "permission denied",
        "xcrun:",
        "license",
        "unable to execute command",
        "invalid value",
        "unknown argument",
    ]
    return any(m in s for m in infra_markers)


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
        tag = (
            "INFRA_COMPILE_ERROR"
            if _is_infra_compile_error(e.stderr)
            else "LLM_COMPILE_ERROR"
        )
        msg = (
            f"\n[{model}] {tag}\n"
            f"Source: {cpp_file}\n"
            f"Compiler cmd: {' '.join(cmd)}\n"
            f"COMPILER STDOUT:\n{e.stdout}\n"
            f"COMPILER STDERR:\n{e.stderr}\n"
        )
        # Most compile failures are because the model produced invalid C++
        if tag == "LLM_COMPILE_ERROR":
            raise LLMCompileError(msg) from None
        raise RuntimeError(msg) from None

    try:
        run = subprocess.run(
            [f"./{bin_file}"], check=True, text=True, capture_output=True
        )
    except subprocess.CalledProcessError as e:
        msg = (
            f"\n[{model}] LLM_RUNTIME_ERROR\n"
            f"Binary: {bin_file}\n"
            f"STDOUT:\n{e.stdout}\n"
            f"STDERR:\n{e.stderr}\n"
        )
        raise LLMRuntimeError(msg) from None

    print(f"\n[{model}] PROGRAM OUTPUT")
    print(run.stdout.rstrip())
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

# --- Models
models = [
    "gpt-5",
    "claude-sonnet-4-5-20250929",
    "qwen2.5-coder",
    "deepseek-coder-v2",
    "gpt-oss:20b",
    "qwen/qwen3-coder-30b-a3b-instruct",
    "openai/gpt-oss-120b",
]


def python_to_cpp_performance(
    models: list = models,
    python: str = pi,
    check_API_key: bool = False,
    ui_launch: bool = False,
):
    # --- Load keys
    load_dotenv(override=True)
    openai_api_key = os.getenv("OPENAI_API_KEY")
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    groq_api_key = os.getenv("GROQ_API_KEY")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")

    if check_API_key:
        if openai_api_key:
            print(f"OpenAI API Key exists and begins {openai_api_key[:8]}")
        else:
            print("OpenAI API Key not set")

        if anthropic_api_key:
            print(f"Anthropic API Key exists and begins {anthropic_api_key[:7]}")
        else:
            print("Anthropic API Key not set")
        if groq_api_key:
            print(f"Groq API Key exists and begins {groq_api_key[:4]}")
        else:
            print("Groq API Key not set (and this is optional)")
        if openrouter_api_key:
            print(f"OpenRouter API Key exists and begins {openrouter_api_key[:6]}")
        else:
            print("OpenRouter API Key not set (and this is optional)")

    # --- Define clients
    openai = OpenAI()

    anthropic_url = "https://api.anthropic.com/v1/"
    anthropic = OpenAI(api_key=anthropic_api_key, base_url=anthropic_url)
    groq_url = "https://api.groq.com/openai/v1"
    groq = OpenAI(api_key=groq_api_key, base_url=groq_url)
    ollama_url = "http://localhost:11434/v1"
    ollama = OpenAI(api_key="ollama", base_url=ollama_url)
    openrouter_url = "https://openrouter.ai/api/v1"
    openrouter = OpenAI(api_key=openrouter_api_key, base_url=openrouter_url)

    clients = {
        "gpt-5": openai,
        "claude-sonnet-4-5-20250929": anthropic,
        "openai/gpt-oss-120b": groq,
        "qwen2.5-coder": ollama,
        "deepseek-coder-v2": ollama,
        "gpt-oss:20b": ollama,
        "qwen/qwen3-coder-30b-a3b-instruct": openrouter,
    }

    # --- Retrieve system info once
    system_info = retrieve_system_info()

    # --- Optional UI
    if ui_launch:
        if gr is None:
            raise RuntimeError(
                "Gradio is not installed. Install with: pip install gradio"
            )

        def convert_code(model: str, python_code: str) -> str:
            client = clients.get(model)
            if client is None:
                raise gr.Error(f"No client configured for model: {model}")
            return port(client, model, python_code, system_info)

        with gr.Blocks() as ui:
            with gr.Row():
                python_box = gr.Code(
                    label="Python code:", lines=28, value=python, language="python"
                )
                cpp_box = gr.Code(label="C++ code:", lines=28, language="cpp")

            with gr.Row():
                model_dd = gr.Dropdown(models, label="Select model", value=models[0])
                convert_btn = gr.Button("Convert code")

            convert_btn.click(
                convert_code, inputs=[model_dd, python_box], outputs=[cpp_box]
            )

        ui.launch(inbrowser=True)
        return None

    # --- Run python benchmark
    py_res, py_time, py_stdout = run_python(python, return_stdout=True)

    print("\n[python] PROGRAM OUTPUT")
    print(py_stdout.rstrip())

    out = {
        "Python result": py_res,
        "Python runtime": py_time,
    }

    for m in models:
        client = clients.get(m)
        if client is None:
            out[f"{m} status"] = "skipped_no_client"
            print(f"Skipping {m}: no client configured")
            continue

        try:
            port(client, m, python, system_info)
            res = compile_and_run(m)

            code_lag = res.split("Result: ")[1]
            code_res = code_lag.split("\nExecution")[0].strip()
            lag = res.split("Time: ")[1]
            runtime_s = lag.split(" seconds")[0].strip()
            performance = py_time / float(runtime_s)

            out[f"{m} status"] = "ok"
            out[f"{m} result"] = code_res
            out[f"{m} runtime"] = runtime_s
            out[f"{m} performance"] = performance

            print(f"[{m}] speedup: {performance:.2f}x (cpp runtime: {runtime_s}s)")

        except LLMCompileError as e:
            out[f"{m} status"] = "llm_compile_error"
            out[f"{m} error"] = str(e)
            print(str(e))
            continue

        except LLMRuntimeError as e:
            out[f"{m} status"] = "llm_runtime_error"
            out[f"{m} error"] = str(e)
            print(str(e))
            continue

        except Exception as e:
            out[f"{m} status"] = "other_error"
            out[f"{m} error"] = str(e)
            print(f"Model failed: {m}\n{e}")
            continue

    return out
