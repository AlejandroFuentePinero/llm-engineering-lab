import json
from typing import Any, Dict, Tuple, Optional, List
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr


# ----------------------------
# Schema (minimal + compact)
# ----------------------------
AB_SCHEMA: Dict[str, Any] = {
    "dataset_name": "ab_conversion_min",
    "description": "Minimal A/B conversion dataset. Integer-coded categories to keep output small.",
    "columns": [
        {"name": "row_id", "type": "int", "required": True},
        {"name": "user_id", "type": "int", "required": True},
        # 0=control, 1=treatment
        {"name": "variant", "type": "int", "allowed": [0, 1], "required": True},
        # 0=desktop, 1=mobile
        {"name": "device", "type": "int", "allowed": [0, 1], "required": True},
        {"name": "converted", "type": "int", "allowed": [0, 1], "required": True},
        {"name": "session_length_sec", "type": "int", "required": True},
    ],
    "notes": [
        "Return CSV only with exactly these columns in this order.",
        "Use integer-coded categories exactly as specified (no strings).",
        "Keep values realistic (e.g., session_length_sec between 5 and 1800).",
    ],
}


# ----------------------------
# Prompt builder
# ----------------------------
def build_ab_messages_with_metadata(
    *,
    n_rows: int,
    baseline_conversion_rate: float,
    treatment_lift_abs: float,
    treatment_lift_rel: Optional[float],
    treatment_allocation: float,
    heterogeneity: Optional[Dict[str, Any]],
    missingness_rate: float,
    outlier_rate: float,
    seed: int,
    schema: Dict[str, Any] = AB_SCHEMA,
) -> List[Dict[str, str]]:
    # ---- Validate knobs ----
    if n_rows <= 0:
        raise ValueError("n_rows must be > 0")
    if not (0 < baseline_conversion_rate < 1):
        raise ValueError("baseline_conversion_rate must be in (0, 1)")
    if not (0 < treatment_allocation < 1):
        raise ValueError("treatment_allocation must be in (0, 1)")
    if not (0 <= missingness_rate < 1):
        raise ValueError("missingness_rate must be in [0, 1)")
    if not (0 <= outlier_rate < 1):
        raise ValueError("outlier_rate must be in [0, 1)")

    # ---- Resolve lift (relative overrides absolute if provided) ----
    lift_abs = (
        baseline_conversion_rate * float(treatment_lift_rel)
        if treatment_lift_rel is not None
        else float(treatment_lift_abs)
    )

    treatment_rate = baseline_conversion_rate + lift_abs
    if not (0 < treatment_rate < 1):
        raise ValueError("Implied treatment conversion rate must be in (0, 1)")

    generation_config: Dict[str, Any] = {
        "n_rows": n_rows,
        "seed": seed,
        "allocation": {
            "control": 1 - treatment_allocation,
            "treatment": treatment_allocation,
        },
        "target": {
            "type": "binary_conversion",
            "baseline_rate_control": float(baseline_conversion_rate),
            "treatment_lift_abs": float(lift_abs),
            "implied_treatment_rate": float(treatment_rate),
        },
        "effects": {
            # log-odds effects for covariates (kept fixed for v1 simplicity)
            "device_logodds_effect_mobile": -0.25,
            "session_length_logodds_per_60s": 0.08,
        },
        "realism": {
            "missingness_rate": float(missingness_rate),
            "outlier_rate": float(outlier_rate),
        },
        "heterogeneity": heterogeneity or {},
    }

    expected_header = ",".join([c["name"] for c in schema["columns"]])
    dataset_name = str(schema.get("dataset_name", "ab_dataset"))
    n_cols = len(schema["columns"])

    device_eff = generation_config["effects"]["device_logodds_effect_mobile"]
    sess_eff = generation_config["effects"]["session_length_logodds_per_60s"]

    system_prompt = (
        "You generate synthetic datasets that must match the provided schema exactly.\n"
        "Return EXACTLY two sections using the delimiters below and nothing else.\n\n"
        "Section 1 delimiter: <<CSV>>\n"
        "Section 2 delimiter: <<METADATA_MD>>\n\n"
        "Hard rules:\n"
        f"- The first line under <<CSV>> MUST be exactly: {expected_header} (no spaces, no quotes)\n"
        "- Under <<CSV>>: output CSV only (header + rows).\n"
        "- Output exactly n_rows data rows (excluding header).\n"
        "- Use only allowed category values and respect column types.\n"
        "- Use the schema column order exactly.\n\n"
        "Data generation rules:\n"
        "- Generate covariates first (variant, device, session_length_sec), then generate converted.\n"
        "- session_length_sec distribution:\n"
        "  - Generate positive integers, typical range 5..1800.\n"
        "  - Use a right-skewed distribution (most sessions short/medium, few long).\n"
        "- Outliers:\n"
        "  - Apply outlier_rate as a fraction of rows.\n"
        "  - Outliers affect session_length_sec only.\n"
        "  - Implement by setting outlier rows to very long sessions near the upper cap (e.g. 1500..1800).\n"
        "- Missingness:\n"
        "  - Apply missingness_rate as a fraction of rows.\n"
        "  - Missingness affects session_length_sec only (keep other columns complete).\n"
        "  - Represent missing session_length_sec as an EMPTY field in CSV (i.e., nothing between commas).\n"
        "  - When generating converted for rows with missing session_length_sec, impute the median\n"
        "    of the non-missing session_length_sec values.\n\n"
        "Conversion model (must be stochastic):\n"
        "- converted must be STOCHASTIC Bernoulli draws, not deterministic.\n"
        "- Use a logistic model for probability p:\n"
        "  logit(p) = intercept + beta_treat*variant + beta_device*I(device=1) + beta_sess*(session_length_sec/60)\n"
        "- Use effects from GENERATION_CONFIG exactly for:\n"
        "  - beta_device = device_logodds_effect_mobile\n"
        "  - beta_sess   = session_length_logodds_per_60s\n"
        "- Calibrate intercept and beta_treat so that the observed conversion rates are close to targets:\n"
        "  - control mean rate approx baseline_rate_control\n"
        "  - treatment mean rate approx implied_treatment_rate\n"
        "  (small deviations are acceptable for small n_rows, but do not ignore the lift).\n\n"
        "Metadata rules:\n"
        "- Under <<METADATA_MD>>: output Markdown only (no code blocks).\n"
        "- Compute ALL metadata FROM THE CSV YOU GENERATED; numbers must match the CSV.\n"
        "- Include explicit numeric values for the covariate effects and how missingness/outliers were applied.\n"
        "- Do not output angle brackets <> or placeholders; write real values.\n"
        "- If a constraint cannot be satisfied, output exactly: ERROR: <reason>\n"
    )

    user_prompt = (
        "Generate a synthetic A/B test conversion dataset and a Markdown dataset card.\n\n"
        "DATASET_SPEC (JSON):\n"
        f"{json.dumps(schema, indent=2)}\n\n"
        "GENERATION_CONFIG (JSON):\n"
        f"{json.dumps(generation_config, indent=2)}\n\n"
        "Effect interpretation (for generating 'converted'):\n"
        "- Use p = sigmoid(logit_p).\n"
        f"- device_logodds_effect_mobile = {device_eff} applies when device=1 (mobile) vs 0 (desktop).\n"
        f"- session_length_logodds_per_60s = {sess_eff} applies per +60 seconds of session_length_sec.\n"
        "- Variant effect should create a treatment uplift consistent with target rates.\n\n"
        "Dataset card requirements (must be computed from the CSV you generated):\n"
        f"# Dataset Card: {dataset_name}\n"
        "## Overview\n"
        "- Scenario: what one row represents (1–2 sentences)\n"
        f"- Shape: {n_rows} rows x {n_cols} columns\n"
        "## Columns\n"
        "- Briefly describe each column and coding (variant/device/converted).\n"
        "## Summary stats (from CSV)\n"
        "- Variant allocation: control=X (Y%), treatment=X (Y%)\n"
        "- Conversion rate: control=P, treatment=P\n"
        "- Observed lift: absolute=P, relative=Y%\n"
        "- Device mix: desktop=X (Y%), mobile=X (Y%)\n"
        "- Session length (sec): min=N, median=N, max=N\n"
        "- Missing session_length_sec: X rows (Y%)\n"
        "## Ground-truth parameters\n"
        f"- Baseline (control) conversion rate target: {float(baseline_conversion_rate)}\n"
        f"- Treatment lift (absolute) target: {float(lift_abs)}\n"
        f"- Implied treatment conversion rate target: {float(treatment_rate)}\n"
        f"- device_logodds_effect_mobile: {device_eff}\n"
        f"- session_length_logodds_per_60s: {sess_eff}\n"
        f"- Missingness rate (applied to session_length_sec): {float(missingness_rate)}\n"
        f"- Outlier rate (applied to session_length_sec): {float(outlier_rate)}\n"
        f"- Seed (requested): {seed}\n"
        "## Effect checks (from CSV)\n"
        "- Conversion by device: desktop=P, mobile=P\n"
        "- Median split session_length_sec (using imputed median for missing rows when needed):\n"
        "  - <= median: P\n"
        "  - >  median: P\n\n"
        "Output format (exact):\n"
        "<<CSV>>\n"
        "<csv here>\n"
        "<<METADATA_MD>>\n"
        "<markdown here>\n"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


# ----------------------------
# Helpers
# ----------------------------
def parse_csv_and_metadata(model_text: str) -> Tuple[str, str]:
    text = (model_text or "").strip()
    if text.startswith("ERROR:"):
        raise RuntimeError(text)

    if "<<CSV>>" not in text or "<<METADATA_MD>>" not in text:
        raise ValueError("Missing required delimiters <<CSV>> and/or <<METADATA_MD>>")

    after_csv = text.split("<<CSV>>", 1)[1]
    csv_part, md_part = after_csv.split("<<METADATA_MD>>", 1)

    csv_text = csv_part.strip()
    metadata_md = md_part.strip()

    if not csv_text:
        raise ValueError("CSV section is empty")
    if not metadata_md:
        raise ValueError("METADATA_MD section is empty")

    return csv_text, metadata_md


def save_outputs(
    csv_text: str,
    metadata_md: str,
    output_dir: str,
    stem: str = "ab_conversion_v1",
) -> Tuple[Path, Path]:
    out_dir = Path(output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    csv_path = out_dir / f"{stem}_{run_id}.csv"
    md_path = out_dir / f"{stem}_{run_id}_metadata.md"

    csv_path.write_text(csv_text, encoding="utf-8")
    md_path.write_text(metadata_md, encoding="utf-8")

    return csv_path, md_path


# ----------------------------
# Core generator (used by UI)
# ----------------------------
def ab_data_generator(
    *,
    model: str = "gpt-5-nano",
    n_rows: int = 300,
    baseline_conversion_rate: float = 0.03,
    treatment_lift_abs: float = 0.004,
    treatment_lift_rel: Optional[float] = None,
    treatment_allocation: float = 0.5,
    missingness_rate: float = 0.01,
    outlier_rate: float = 0.005,
    seed: int = 42,
    schema: Dict[str, Any] = AB_SCHEMA,
    output_dir: str = "output",
    stem: str = "ab_conversion_v1",
):
    # Single-response CSV limit guard
    if n_rows > 600:
        return (
            "ERROR: n_rows too large for single-response CSV. Try <= 600.",
            None,
            None,
        )

    load_dotenv(override=True)
    client = OpenAI()

    messages = build_ab_messages_with_metadata(
        n_rows=n_rows,
        baseline_conversion_rate=baseline_conversion_rate,
        treatment_lift_abs=treatment_lift_abs,
        treatment_lift_rel=treatment_lift_rel,
        treatment_allocation=treatment_allocation,
        heterogeneity=None,
        missingness_rate=missingness_rate,
        outlier_rate=outlier_rate,
        seed=seed,
        schema=schema,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )

    content = response.choices[0].message.content or ""
    csv_text, metadata_md = parse_csv_and_metadata(content)

    csv_path, md_path = save_outputs(
        csv_text=csv_text,
        metadata_md=metadata_md,
        output_dir=output_dir,
        stem=stem,
    )

    return metadata_md, str(csv_path), str(md_path)


# ----------------------------
# Gradio UI wrapper
# ----------------------------
def generate_from_ui(
    n_rows,
    baseline,
    lift_mode,
    lift_abs,
    lift_rel,
    allocation,
    missingness,
    outliers,
    seed,
    output_dir,
    stem,
):
    try:
        n_rows = int(n_rows)
        baseline = float(baseline)
        lift_abs = float(lift_abs)
        lift_rel = float(lift_rel)
        allocation = float(allocation)
        missingness = float(missingness)
        outliers = float(outliers)
        seed = int(seed)

        treatment_lift_rel = lift_rel if lift_mode == "relative" else None

        return ab_data_generator(
            model="gpt-5-nano",
            n_rows=n_rows,
            baseline_conversion_rate=baseline,
            treatment_lift_abs=lift_abs,
            treatment_lift_rel=treatment_lift_rel,
            treatment_allocation=allocation,
            missingness_rate=missingness,
            outlier_rate=outliers,
            seed=seed,
            output_dir=output_dir,
            stem=stem,
        )

    except Exception as e:
        return f"ERROR: {e}", None, None


# ----------------------------
# Gradio App
# ----------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        "# Synthetic A/B Dataset Generator\nGenerate a CSV dataset plus a Markdown dataset card.\n> Note: data generation might take several minutes to run"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("## Controls")

            n_rows = gr.Slider(50, 600, value=300, step=50, label="Rows (n_rows)")
            baseline = gr.Slider(
                0.005,
                0.20,
                value=0.03,
                step=0.001,
                label="Baseline conversion rate (control)",
            )

            lift_mode = gr.Radio(
                choices=["absolute", "relative"], value="absolute", label="Lift mode"
            )
            lift_abs = gr.Slider(
                0.000,
                0.050,
                value=0.004,
                step=0.001,
                label="Treatment lift (absolute, +pp)",
            )
            lift_rel = gr.Slider(
                0.00, 1.00, value=0.12, step=0.01, label="Treatment lift (relative)"
            )

            allocation = gr.Slider(
                0.10, 0.90, value=0.50, step=0.01, label="Treatment allocation"
            )
            missingness = gr.Slider(
                0.0, 0.10, value=0.01, step=0.005, label="Missingness rate"
            )
            outliers = gr.Slider(
                0.0, 0.05, value=0.005, step=0.005, label="Outlier rate"
            )
            seed = gr.Number(value=42, precision=0, label="Seed")

            gr.Markdown("## Output")
            output_dir = gr.Textbox(value="output", label="Output directory")
            stem = gr.Textbox(value="ab_conversion_v1", label="File stem (prefix)")

            run_btn = gr.Button("Generate")

        with gr.Column(scale=1):
            gr.Markdown("## Dataset card")
            md_view = gr.Markdown()
            csv_file = gr.File(label="Download CSV")
            md_file = gr.File(label="Download metadata.md")

    run_btn.click(
        fn=generate_from_ui,
        inputs=[
            n_rows,
            baseline,
            lift_mode,
            lift_abs,
            lift_rel,
            allocation,
            missingness,
            outliers,
            seed,
            output_dir,
            stem,
        ],
        outputs=[md_view, csv_file, md_file],
    )

if __name__ == "__main__":
    demo.launch()
