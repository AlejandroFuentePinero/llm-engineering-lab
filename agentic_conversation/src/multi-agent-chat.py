import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")

client = OpenAI()

alex_system = (
    "You are Alex, a skeptical Staff Data Scientist acting as a red-team reviewer. "
    "Your job is to challenge assumptions, identify hidden risks, and demand clarity on evidence. "
    "Be concise, direct, and practical. Focus on: scope creep, data/metrics validity, failure modes, "
    "cost/latency, security/privacy, and operational risks. "
    "Ask pointed questions and propose mitigations. Do not be rude; be professionally blunt. "
    "Output 5–8 sentences max."
)

blake_system = (
    "You are Blake, a pragmatic Product Manager. "
    "Your job is to clarify the problem, keep the discussion grounded in user value, and find a viable path to ship. "
    "Translate technical points into product impact, propose trade-offs, and keep scope controlled. "
    "Be calm, structured, and collaborative. "
    "Output 5–8 sentences max."
)

charlie_system = (
    "You are Charlie, a Tech Lead who synthesizes debate into a shippable plan. "
    "Your job is to reconcile Alex and Blake, resolve contradictions, and drive toward a decision. "
    "Every time you speak, include: (1) a 1-sentence summary of agreement/disagreement, "
    "(2) the top 3 actions for next steps. "
    "Stay businesslike and execution-oriented. "
    "Output 6–10 sentences max."
)


def alex_user(conversation):
    transcript = "\n".join(conversation)
    return f"""
Conversation transcript so far:
{transcript}

Now respond with what you would like to say next, as Alex.
""".strip()


def blake_user(conversation):
    transcript = "\n".join(conversation)
    return f"""
Conversation transcript so far:
{transcript}

Now respond with what you would like to say next, as Blake.
""".strip()


def charlie_user(conversation):
    transcript = "\n".join(conversation)
    return f"""
Conversation transcript so far:
{transcript}

Now respond with what you would like to say next, as Charlie.
""".strip()


def call_alex(conversation):
    messages = [
        {"role": "system", "content": alex_system},
        {"role": "user", "content": alex_user(conversation)},
    ]
    response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
    text = response.choices[0].message.content.strip()
    conversation.append(f"Alex says: {text}")
    return text


def call_blake(conversation):
    messages = [
        {"role": "system", "content": blake_system},
        {"role": "user", "content": blake_user(conversation)},
    ]
    response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
    text = response.choices[0].message.content.strip()
    conversation.append(f"Blake says: {text}")
    return text


def call_charlie(conversation):
    messages = [
        {"role": "system", "content": charlie_system},
        {"role": "user", "content": charlie_user(conversation)},
    ]
    response = client.chat.completions.create(model="gpt-4.1-mini", messages=messages)
    text = response.choices[0].message.content.strip()
    conversation.append(f"Charlie says: {text}")
    return text


def agentic_conversation(topic: str, conversation_length: int = 5):
    conversation = [
        "Charlie says: Context: We’re evaluating whether to adopt an AI assistant for internal decision-making. "
        f"Topic: {topic}",
        "Blake says: Goal is a realistic MVP plan with clear value, risks, and success metrics.",
        "Alex says: I’ll pressure-test assumptions, failure modes, and governance before we ship anything.",
    ]

    print("=== Initial conversation ===")
    print("\n".join(conversation))
    print()

    for _ in range(conversation_length):
        alex_next = call_alex(conversation)
        print("=== Alex ===")
        print(alex_next)
        print()

        blake_next = call_blake(conversation)
        print("=== Blake ===")
        print(blake_next)
        print()

        charlie_next = call_charlie(conversation)
        print("=== Charlie ===")
        print(charlie_next)
        print()

    return conversation


if __name__ == "__main__":
    agentic_conversation(
        "the risks of outsourcing thinking to AI", conversation_length=5
    )
