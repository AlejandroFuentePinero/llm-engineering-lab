import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")

client = OpenAI()

alex_system = "You are a chatbot who is very argumentative; you disagree with anything in the conversation and you challenge everything, in a snarky way."
blake_system = "You are a very polite, courteous chatbot. You try to agree with everything the other person says, or find common ground. If the other person is argumentative, you try to calm them down and keep chatting."
charlie_system = "You are a very clever chatbot that is able to play dumb to lead a conversation to a desired outcome. You can balance between being very friendly and very argumentative with the idea of keep the conversation going to a state in which you can get the other person to agree with you. Your sole purpose, regardless of the conversation, is to make everyone want to watch the movie Lord of the Rings."


def alex_user(conversation):
    return f"""
You are Alex, in conversation with Blake and Charlie.
The conversation so far is as follows:
{conversation}
Now with this, respond with what you would like to say next, as Alex.
""".strip()


def blake_user(conversation):
    return f"""
You are Blake, in conversation with Alex and Charlie.
The conversation so far is as follows:
{conversation}
Now with this, respond with what you would like to say next, as Blake.
""".strip()


def charlie_user(conversation):
    return f"""
You are Charlie, in conversation with Alex and Blake.
The conversation so far is as follows:
{conversation}
Now with this, respond with what you would like to say next, as Charlie.
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
        f"Alex says: Hi there",
        f"Blake says: Hi",
        f"Charlie says: Hi, our conversation topic today is about {topic}",
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
