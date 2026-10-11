from submission.anonymizer.base_prompt import compose
from submission.anonymizer.categories import Category
from submission.anonymizer.llm import LLMClient
from submission.anonymizer.schemas import DetectorResult, Finding


def format_category(category: Category) -> str:
    examples = "; ".join(category.examples)
    do_not_mask = "; ".join(category.do_not_mask)
    return (
        f"Categoria: {category.id} ({category.name})\n"
        f"Descrição: {category.description}\n"
        f"Exemplos: {examples}\n"
        f"Não anonimizar: {do_not_mask}"
    )


def format_categories(categories: list[Category]) -> str:
    return "\n\n".join(format_category(c) for c in categories)


def detect(text: str, categories: list[Category], base: str,
           client: LLMClient) -> list[Finding]:
    """Find sensitive data of every category in a single call.

    One call instead of one per category keeps a document within free-tier
    rate limits; the category the model assigns is only a report label,
    masking doesn't depend on it.
    """
    prompt = compose(base, "detector", text,
                     category_block=format_categories(categories))
    return client.generate_json(prompt, DetectorResult).findings
