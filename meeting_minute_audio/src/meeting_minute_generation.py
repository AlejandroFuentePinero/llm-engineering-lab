import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

try:
    from IPython.display import Markdown, display
except Exception:
    Markdown = None
    display = None


LOCAL_AUDIO_DEMO = Path(
    "/Users/alejandrofp/Desktop/Projects/01_Course_projects/AI/llm-projects/meeting_minute_audio/data/denver_extract.mp3"
)


def messages_generator(transcription: str) -> List[Dict[str, str]]:
    system_message = (
        "You write meeting minutes from transcripts.\n"
        "Output must be Markdown (no code blocks).\n"
        "Be faithful to the transcript: do NOT invent names, dates, locations, decisions, votes, or action items.\n"
        "If attendees/date/location/owners are not explicitly stated, write 'Not specified'.\n"
        "Prefer clarity and structure over verbosity.\n"
    )

    user_prompt = (
        "Below is an extract transcript of a Denver council meeting.\n\n"
        "Produce minutes in Markdown (no code blocks) with this exact structure and level of detail:\n\n"
        "1) Summary\n"
        "- Date: ... (or 'Not specified')\n"
        "- Location: ... (or 'Not specified')\n"
        "- Attendees: ... (or 'Not specified')\n"
        "- 3–5 sentence overview of what happened\n\n"
        "2) Key discussion points (8–12 bullets)\n"
        "- Each bullet: 1–2 sentences, include specific topics/positions mentioned.\n\n"
        "3) Takeaways (3–6 bullets)\n"
        "- Only items supported by the transcript.\n\n"
        "4) Action items (0–8 bullets)\n"
        "- Format: Action — Owner — Due date (if stated)\n"
        "- If no clear owner, Owner: Not specified\n"
        "- If none stated, write: 'No action items explicitly stated.'\n\n"
        "Transcription:\n"
        f"{transcription}"
    )

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt},
    ]


def meeting_minute_generator(
    audio_filename: str | Path = LOCAL_AUDIO_DEMO,
    summary_model: str = "gpt-4.1-mini",
    audio_model: str = "gpt-4o-mini-transcribe",
    show_audio_transcription: bool = False,
    max_tokens: int = 2000,
    temperature: float = 0.1,
    render_markdown: bool = True,
    save_transcript: bool = False,
    save_transcript_dir: str | None = "data/output/transcripts",
) -> str:
    load_dotenv(override=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to your environment or .env file."
        )

    client = OpenAI(api_key=api_key)

    audio_path = Path(audio_filename)
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with audio_path.open("rb") as audio_file:
        transcription: str = client.audio.transcriptions.create(
            model=audio_model,
            file=audio_file,
            response_format="text",
        )

    if show_audio_transcription:
        print(transcription)

    if save_transcript:
        if not save_transcript_dir:
            raise ValueError(
                "save_transcript_dir must be set when save_transcript=True"
            )

        out_dir = Path(save_transcript_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_path = out_dir / f"{audio_path.stem}_{ts}.txt"
        out_path.write_text(transcription, encoding="utf-8")
        print(f"Saved transcript to: {out_path}")

    response = client.chat.completions.create(
        model=summary_model,
        messages=messages_generator(transcription),
        max_tokens=max_tokens,
        temperature=temperature,
    )

    minutes_md = response.choices[0].message.content or ""

    if render_markdown and display is not None and Markdown is not None:
        display(Markdown(minutes_md))
    else:
        print(minutes_md)

    return minutes_md


if __name__ == "__main__":
    md = meeting_minute_generator(
        audio_filename="/Users/alejandrofp/Desktop/Projects/01_Course_projects/AI/llm-projects/meeting_minute_audio/data/denver_extract.mp3",
        show_audio_transcription=False,
    )
    print(md)
