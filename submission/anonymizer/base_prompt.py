import re
from pathlib import Path
from string import Template

from submission.anonymizer.config import PROMPT_PATH

MARKER = "### DOCUMENTO ###"
PROMPTS_DIR = Path(__file__).with_name("prompts")


def load_base(path: Path = PROMPT_PATH) -> str:
    """Return prompt.txt's instructions, without the embedded document.

    prompt.txt carries the document so it works single-shot in CI; the
    pipeline passes the document to each agent itself.
    """
    text = path.read_text(encoding="utf-8")
    # Only a line holding just the marker splits: the instructions may
    # mention the marker inline when telling the model where to look.
    match = re.search(rf"^{re.escape(MARKER)}$", text, flags=re.MULTILINE)
    if match:
        text = text[:match.start()]
    return text.strip()


def compose(base: str, agent: str, document: str, **fields: str) -> str:
    # string.Template ($name) instead of str.format, so the JSON examples
    # in the templates need no brace escaping.
    template = Template(
        (PROMPTS_DIR / f"{agent}.md").read_text(encoding="utf-8")
    )
    focus = template.substitute(fields).strip()
    return f"{base}\n\n{focus}\n\n{MARKER}\n{document}"
