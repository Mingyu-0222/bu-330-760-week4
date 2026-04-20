"""Math agent that solves questions using tools in a ReAct loop."""

import json

from dotenv import load_dotenv
from pydantic_ai import Agent
from calculator import calculate

load_dotenv()

# Configure your model below. Examples:
#   "google-gla:gemini-2.5-flash"       (needs GOOGLE_API_KEY)
#   "openai:gpt-4o-mini"                (needs OPENAI_API_KEY)
#   "anthropic:claude-sonnet-4-6"       (needs ANTHROPIC_API_KEY)
MODEL = "openai:gpt-4o-mini"

agent = Agent(
    MODEL,
    system_prompt=(
        "You are a helpful assistant. Solve each question step by step. "
        "Use the calculator tool for arithmetic. "
        "Use the product_lookup tool whenever a question mentions products from the catalog. "
        "Never guess a product price. Always call product_lookup first for catalog products, "
        "then use the calculator tool for any arithmetic based on those prices. "
        "If a question cannot be answered with the information given, say so."
    ),
)


@agent.tool_plain
def calculator_tool(expression: str) -> str:
    """Evaluate a math expression and return the result.

    Examples: "847 * 293", "10000 * (1.07 ** 5)", "23 % 4"
    """
    return calculate(expression)


@agent.tool_plain
def product_lookup(product_name: str) -> str:
    """Look up the price of a product by name.
    Use this when a question asks about product prices from the catalog.
    """
    with open("products.json", "r", encoding="utf-8") as f:
        catalog = json.load(f)

    target = product_name.strip().lower()

    # Case 1: products.json is a dictionary like:
    # {"Alpha Widget": 25, "Beta Widget": 40}
    if isinstance(catalog, dict):
        for name, price in catalog.items():
            if str(name).strip().lower() == target:
                return str(price)

        available_products = ", ".join(sorted(str(name) for name in catalog.keys()))
        return f"Product not found. Available products: {available_products}"

    # Case 2: products.json is a list like:
    # [{"name": "Alpha Widget", "price": 25}, ...]
    if isinstance(catalog, list):
        available_names = []

        for item in catalog:
            if not isinstance(item, dict):
                continue

            name = item.get("name") or item.get("product_name") or item.get("product")
            price = item.get("price")

            if name is not None:
                available_names.append(str(name))
                if str(name).strip().lower() == target and price is not None:
                    return str(price)

        available_products = ", ".join(sorted(available_names))
        return f"Product not found. Available products: {available_products}"

    return "Product catalog format is invalid."


def load_questions(path: str = "math_questions.md") -> list[str]:
    """Load numbered questions from the markdown file."""
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and line[0].isdigit() and ". " in line[:4]:
                questions.append(line.split(". ", 1)[1])
    return questions


def main():
    questions = load_questions()
    for i, question in enumerate(questions, 1):
        print(f"## Question {i}")
        print(f"> {question}\n")

        result = agent.run_sync(question)

        print("### Trace")
        for message in result.all_messages():
            for part in message.parts:
                kind = part.part_kind
                if kind in ("user-prompt", "system-prompt"):
                    continue
                elif kind == "text":
                    print(f"- **Reason:** {part.content}")
                elif kind == "tool-call":
                    print(f"- **Act:** `{part.tool_name}({part.args})`")
                elif kind == "tool-return":
                    print(f"- **Result:** `{part.content}`")

        print(f"\n**Answer:** {result.output}\n")
        print("---\n")


if __name__ == "__main__":
    main()