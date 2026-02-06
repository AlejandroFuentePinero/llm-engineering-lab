import os
import json
from dotenv import load_dotenv
from openai import OpenAI
import gradio as gr
import sqlite3
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime, timedelta

DB = "prices.db"

# -----------------------
# DB init (must run before seeding)
# -----------------------


def init_db():
    with sqlite3.connect(DB) as conn:
        c = conn.cursor()

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS prices (
            city TEXT PRIMARY KEY,
            price REAL
        )
        """
        )

        c.execute(
            """
        CREATE TABLE IF NOT EXISTS bookings (
            booking_id INTEGER PRIMARY KEY AUTOINCREMENT,
            city TEXT NOT NULL,
            depart_at TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
        )

        conn.commit()


# -----------------------
# Tool - get_ticket_price
# -----------------------


def get_ticket_price(city):
    print(f"DATABASE TOOL CALLED: Getting price for {city}", flush=True)
    if not city:
        return "No price data available for this city"
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT price FROM prices WHERE city = ?", (city.lower(),))
        result = cursor.fetchone()
        return (
            f"Ticket price to {city} is ${result[0]}"
            if result
            else "No price data available for this city"
        )


# -----------------------
# Tool - book_ticket
# -----------------------


def book_ticket(destination_city: str, depart_at: str | None = None) -> str:
    """
    Create a mock booking. If depart_at is omitted, default to tomorrow at 09:00.
    depart_at should be a simple string like '2026-02-07 09:00'.
    """
    city = (destination_city or "").strip().lower()
    if not city:
        return "Booking failed: destination city is missing."

    if depart_at is None or not str(depart_at).strip():
        dt = datetime.now() + timedelta(days=1)
        depart_at = dt.strftime("%Y-%m-%d 09:00")

    with sqlite3.connect(DB) as conn:
        c = conn.cursor()
        c.execute(
            "INSERT INTO bookings (city, depart_at) VALUES (?, ?)",
            (city, str(depart_at)),
        )
        booking_id = c.lastrowid
        conn.commit()

    return f"Booked. Booking ID {booking_id}. Destination {city.title()}. Departure {depart_at}."


# -----------------------
# Tool schemas
# -----------------------

book_function = {
    "name": "book_ticket",
    "description": "Create a mock booking for a flight ticket to a destination city at a chosen departure time.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "City to book the ticket to (e.g., 'Tokyo').",
            },
            "depart_at": {
                "type": "string",
                "description": "Departure date/time as a simple string, e.g. '2026-02-07 09:00'. If omitted, choose a reasonable default.",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

price_function = {
    "name": "get_ticket_price",
    "description": "Get the price of a return ticket to the destination city.",
    "parameters": {
        "type": "object",
        "properties": {
            "destination_city": {
                "type": "string",
                "description": "The city that the customer wants to travel to",
            },
        },
        "required": ["destination_city"],
        "additionalProperties": False,
    },
}

tools = [
    {"type": "function", "function": price_function},
    {"type": "function", "function": book_function},
]

# -----------------------
# Tool router
# -----------------------


def handle_tool_calls(message):
    responses = []
    cities = []

    for tool_call in message.tool_calls:
        name = tool_call.function.name

        try:
            arguments = json.loads(tool_call.function.arguments)
        except Exception:
            arguments = {}

        if name == "get_ticket_price":
            city = arguments.get("destination_city")
            if city:
                cities.append(city)
            out = get_ticket_price(city)

        elif name == "book_ticket":
            city = arguments.get("destination_city")
            if city:
                cities.append(city)
            depart_at = arguments.get("depart_at")
            out = book_ticket(city, depart_at)

        else:
            out = f"Tool error: unknown tool '{name}'."

        responses.append({"role": "tool", "content": out, "tool_call_id": tool_call.id})

    return responses, cities


# -----------------------
# System prompt
# -----------------------

system_message = """
You are FlightAI. Keep answers to one sentence.
If the user asks to book, confirm first: ask if they want you to proceed and request a departure time if missing.
Only call book_ticket after the user explicitly says yes.
"""

# -----------------------
# SQL helper to populate DB
# -----------------------


def set_ticket_price(city, price):
    with sqlite3.connect(DB) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO prices (city, price) VALUES (?, ?) "
            "ON CONFLICT(city) DO UPDATE SET price = ?",
            (city.lower(), price, price),
        )
        conn.commit()


# Ensure tables exist BEFORE seeding
init_db()

ticket_prices = {"london": 799, "paris": 899, "tokyo": 1420, "sydney": 2999}
for city, price in ticket_prices.items():
    set_ticket_price(city, price)

# -----------------------
# Helper - generate image
# -----------------------


def artist(city, openai: OpenAI):
    image_response = openai.images.generate(
        model="dall-e-3",
        prompt=(
            f"An image representing a vacation in {city}, showing tourist spots "
            f"and everything unique about {city}, in a vibrant pop-art style"
        ),
        size="1024x1024",
        n=1,
        response_format="b64_json",
    )
    image_base64 = image_response.data[0].b64_json
    image_data = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_data))


def talker(message, openai: OpenAI):
    response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=message,
    )
    return response.content


# -----------------------
# Main function (agent)
# -----------------------


def booking_agent(history):
    load_dotenv(override=True)
    _ = os.getenv("OPENAI_API_KEY")
    MODEL = "gpt-4.1-mini"
    openai = OpenAI()

    # Ensure DB exists (safe to call repeatedly)
    init_db()

    history = [{"role": h["role"], "content": h["content"]} for h in (history or [])]
    messages = [{"role": "system", "content": system_message}] + history

    response = openai.chat.completions.create(
        model=MODEL, messages=messages, tools=tools
    )

    cities = []
    image = None

    while response.choices[0].finish_reason == "tool_calls":
        message = response.choices[0].message

        tool_responses, new_cities = handle_tool_calls(message)
        cities.extend(new_cities)

        messages.append(message)
        messages.extend(tool_responses)

        response = openai.chat.completions.create(
            model=MODEL, messages=messages, tools=tools
        )

    reply = response.choices[0].message.content or ""
    history = history + [{"role": "assistant", "content": reply}]

    voice = talker(reply, openai)

    if cities:
        image = artist(cities[0], openai)

    return history, voice, image


# -----------------------
# Gradio Blocks UI wrapper
# -----------------------


def put_message_in_chatbot(message, history):
    history = history or []
    history = history + [{"role": "user", "content": message}]
    return "", history


def chat(history):
    return booking_agent(history)


with gr.Blocks() as ui:
    with gr.Row():
        chatbot = gr.Chatbot(height=500, type="messages")
        image_output = gr.Image(height=500, interactive=False)
    with gr.Row():
        audio_output = gr.Audio(autoplay=True)
    with gr.Row():
        message = gr.Textbox(label="Chat with our AI Assistant:")

    message.submit(
        put_message_in_chatbot,
        inputs=[message, chatbot],
        outputs=[message, chatbot],
    ).then(
        chat,
        inputs=chatbot,
        outputs=[chatbot, audio_output, image_output],
    )

ui.launch(inbrowser=True)
