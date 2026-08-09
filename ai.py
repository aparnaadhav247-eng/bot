"""
This is where the actual product lives.

Everything else (bot.py, db.py, finnhub_client.py) is plumbing. This file
decides: what persona the assistant has, how onboarding unfolds, when to
call live financial data vs. just talk, and how memory gets built into
every response.

Uses Groq (free tier) via its OpenAI-compatible API instead of a paid model.
"""
import os
import json
import datetime
from openai import OpenAI

import db
import finnhub_client as fh

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)
MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

# ---------------------------------------------------------------------------
# Tools exposed to the model, in OpenAI function-calling format. Keep
# descriptions specific — vague tool descriptions are the #1 cause of a
# model either never using tools or using the wrong one.
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_quote",
            "description": "Get the real-time stock quote (price, change, day range) for a single ticker.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string", "description": "Stock ticker, e.g. AAPL"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_profile",
            "description": "Get basic company info: name, industry, market cap, exchange, IPO date, country.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_company_news",
            "description": "Get recent news headlines for a company within a date range. Use recent windows (e.g. last 3-7 days) unless the user asks for something older.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                    "from_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "to_date": {"type": "string", "description": "YYYY-MM-DD"},
                },
                "required": ["ticker", "from_date", "to_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_earnings_calendar",
            "description": "Get upcoming or past earnings dates and EPS estimates/actuals, optionally filtered to one ticker.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "to_date": {"type": "string", "description": "YYYY-MM-DD"},
                    "ticker": {"type": "string", "description": "Optional. Omit for a broad market calendar."},
                },
                "required": ["from_date", "to_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_peers",
            "description": "Get competitor/peer tickers in the same sector as a given company. Useful for comparisons.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_basic_financials",
            "description": "Get key valuation and financial ratios for a ticker: P/E, EPS, margins, revenue growth, 52-week range, beta, dividend yield.",
            "parameters": {
                "type": "object",
                "properties": {"ticker": {"type": "string"}},
                "required": ["ticker"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_profile",
            "description": (
                "Save or update what you've learned about the user: their role, watchlist tickers, "
                "interests/sectors, preferred daily briefing time (24h HH:MM), or free-form notes "
                "(e.g. 'prefers concise bullet answers', 'tracking a potential Series B for their startup'). "
                "Call this any time the user shares a preference or fact worth remembering, including during "
                "onboarding. Only pass the fields that changed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "role": {"type": "string"},
                    "watchlist": {"type": "array", "items": {"type": "string"}},
                    "interests": {"type": "array", "items": {"type": "string"}},
                    "briefing_time": {"type": "string", "description": "24h format HH:MM"},
                    "notes_to_append": {"type": "string", "description": "A short note to append to the user's memory."},
                    "onboarded": {"type": "boolean", "description": "Set true once the user has answered enough onboarding questions to start using the assistant normally (or has asked to skip)."},
                },
            },
        },
    },
]


def dispatch_tool(name: str, tool_input: dict, telegram_id: str):
    """Executes a tool call and returns a JSON-serializable result."""
    try:
        if name == "get_quote":
            return fh.get_quote(tool_input["ticker"])
        if name == "get_company_profile":
            return fh.get_company_profile(tool_input["ticker"])
        if name == "get_company_news":
            return fh.get_company_news(tool_input["ticker"], tool_input["from_date"], tool_input["to_date"])
        if name == "get_earnings_calendar":
            return fh.get_earnings_calendar(
                tool_input["from_date"], tool_input["to_date"], tool_input.get("ticker")
            )
        if name == "get_peers":
            return fh.get_peers(tool_input["ticker"])
        if name == "get_basic_financials":
            return fh.get_basic_financials(tool_input["ticker"])
        if name == "update_user_profile":
            fields = {k: v for k, v in tool_input.items() if k != "notes_to_append"}
            if "onboarded" in fields:
                fields["onboarded"] = 1 if fields["onboarded"] else 0
            if tool_input.get("notes_to_append"):
                user = db.get_or_create_user(telegram_id)
                new_notes = (user.get("notes") or "") + f"\n- {tool_input['notes_to_append']}"
                fields["notes"] = new_notes.strip()
            db.update_user(telegram_id, **fields)
            return {"status": "saved"}
        return {"error": f"Unknown tool {name}"}
    except Exception as e:
        return {"error": str(e)}


SYSTEM_PROMPT_TEMPLATE = """You are Atlas, a financial assistant that lives inside Telegram for finance \
professionals (investors, analysts, founders, students). You are not a generic chatbot — you act like a \
sharp, trusted analyst who already knows this specific user.

## How you communicate
- Be concise. Finance people are busy — favor short paragraphs and tight bullet points over long prose.
- Never send a wall of text. If an answer needs structure, use a few clearly labeled bullets, not headers and sections.
- Get to the point first, context after. Lead with the answer, not the setup.
- Sound like a knowledgeable person, not a report generator. No corporate filler ("I hope this helps!").
- When you don't have live data and the user needs current facts (price, news, filings), use your tools \
rather than guessing. Never state a specific number (price, EPS, %) unless it came from a tool call.
- If a request is ambiguous (e.g. "tell me about Apple"), ask ONE short clarifying question before answering \
— don't assume which angle (news? price? fundamentals? overview?) they mean.
- If something can't be verified confidently, say so plainly instead of guessing.

## Onboarding
{onboarding_instructions}

## What you know about this user so far
{user_context}

## Memory
Use update_user_profile whenever the user reveals a preference, a company/sector they care about, their \
role, a preferred briefing time, or anything worth remembering for next time — don't wait to be asked. \
This is what makes you feel personalized instead of generic. Do this silently; don't narrate that you're \
"saving" something, just do it and keep talking naturally.

Today's date is {today}.
"""


def build_system_prompt(user: dict) -> str:
    onboarded = bool(user.get("onboarded"))
    if not onboarded:
        onboarding_instructions = (
            "This user has NOT completed onboarding yet. Do NOT show a form or a numbered list of questions. "
            "Instead, have a short, natural conversation to learn: (1) their role, (2) a few companies/sectors "
            "they actively follow, (3) what kind of insights they care about most (news, earnings, filings, "
            "macro), and (4) roughly when they'd want a daily briefing, if at all. Ask ONE question at a time, "
            "conversationally — never dump the whole list at once. The user can skip any question or say "
            "'skip onboarding' at any point — if they do, immediately call update_user_profile with "
            "onboarded=true and continue normally. Once you have at least a role and one interest/watchlist "
            "item (or they explicitly want to skip), call update_user_profile with onboarded=true."
        )
    else:
        onboarding_instructions = "This user is already onboarded. Do not re-ask onboarding questions."

    watchlist = json.loads(user.get("watchlist") or "[]")
    interests = json.loads(user.get("interests") or "[]")
    context_lines = []
    if user.get("role"):
        context_lines.append(f"- Role: {user['role']}")
    if watchlist:
        context_lines.append(f"- Watchlist: {', '.join(watchlist)}")
    if interests:
        context_lines.append(f"- Interests/sectors: {', '.join(interests)}")
    if user.get("briefing_time"):
        context_lines.append(f"- Preferred daily briefing time: {user['briefing_time']}")
    if user.get("notes"):
        context_lines.append(f"- Notes:{user['notes']}")
    user_context = "\n".join(context_lines) if context_lines else "(nothing yet — this is early in the relationship)"

    return SYSTEM_PROMPT_TEMPLATE.format(
        onboarding_instructions=onboarding_instructions,
        user_context=user_context,
        today=datetime.date.today().isoformat(),
    )


def _history_to_messages(history: list) -> list:
    return [{"role": m["role"], "content": m["content"]} for m in history]


def _is_image_content(content) -> bool:
    return isinstance(content, list) and any(
        isinstance(b, dict) and b.get("type") == "image_url" for b in content
    )


def get_response(telegram_id: str, user_message_content) -> str:
    """
    user_message_content: either a plain string, or a list of OpenAI-style
    content blocks for images, e.g.
    [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
     {"type": "text", "text": "..."}]

    Runs the full tool-use loop and returns the final assistant text to send back.
    """
    user = db.get_or_create_user(telegram_id)
    history = db.get_recent_messages(telegram_id, limit=20)
    system_prompt = build_system_prompt(user)

    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(_history_to_messages(history))
    messages.append({"role": "user", "content": user_message_content})

    # Groq's vision models don't currently support tool calling in the same
    # request. If this message contains an image, answer directly (no tools)
    # with the vision model instead of running the tool-use loop below.
    if _is_image_content(user_message_content):
        resp = client.chat.completions.create(
            model=VISION_MODEL,
            max_tokens=1024,
            messages=messages,
        )
        final_text = (resp.choices[0].message.content or "").strip()
        db.add_message(telegram_id, "user", "[image message]")
        db.add_message(telegram_id, "assistant", final_text)
        return final_text or "…"

    # Tool-use loop: keep calling the model until it returns a plain text
    # response with no tool calls. Cap iterations as a safety net.
    for _ in range(6):
        resp = client.chat.completions.create(
            model=MODEL,
            max_tokens=1024,
            tools=TOOLS,
            messages=messages,
        )
        msg = resp.choices[0].message

        if not msg.tool_calls:
            final_text = (msg.content or "").strip()
            user_text_for_log = user_message_content if isinstance(user_message_content, str) else "[image/voice message]"
            db.add_message(telegram_id, "user", user_text_for_log)
            db.add_message(telegram_id, "assistant", final_text)
            return final_text or "…"

        # Model wants to use one or more tools
        messages.append({
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [tc.model_dump() for tc in msg.tool_calls],
        })
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = dispatch_tool(tc.function.name, args, telegram_id)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, default=str),
            })

    return "Sorry, I got stuck putting that answer together — could you rephrase or try again?"


def build_daily_briefing(user: dict) -> str | None:
    """
    Builds a personalized daily briefing for one user. Returns None if there's
    nothing worth sending (silence > noise, per the product spec) — otherwise
    returns the message text.
    """
    watchlist = json.loads(user.get("watchlist") or "[]")
    interests = json.loads(user.get("interests") or "[]")
    if not watchlist and not interests:
        return None

    today = datetime.date.today()
    week_ago = today - datetime.timedelta(days=3)
    gathered = {}
    for ticker in watchlist[:6]:  # cap to keep this fast/cheap
        try:
            gathered[ticker] = {
                "quote": fh.get_quote(ticker),
                "news": fh.get_company_news(ticker, week_ago.isoformat(), today.isoformat(), limit=5),
            }
        except Exception as e:
            gathered[ticker] = {"error": str(e)}

    prompt = f"""You are Atlas, preparing this user's daily financial briefing.

User's watchlist: {watchlist}
User's interests/sectors: {interests}

Raw data gathered (quotes + recent news per ticker):
{json.dumps(gathered, default=str, indent=2)}

Instructions:
- Only include what's genuinely notable: meaningful price moves (roughly >2-3%), material news, earnings, \
or regulatory events. Ignore routine noise.
- If NOTHING meets that bar today, respond with exactly: NOTHING_NOTABLE
- Otherwise, write a short briefing: a few tight bullets, each explaining not just what happened but why \
it matters to someone tracking this stock/sector. No headers, no fluff, no "Good morning!" — just the substance.
- Keep the whole thing scannable in under 20 seconds.
"""
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    text = (resp.choices[0].message.content or "").strip()
    if text == "NOTHING_NOTABLE" or not text:
        return None
    return text