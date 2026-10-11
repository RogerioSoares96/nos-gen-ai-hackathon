from pydantic import BaseModel, Field


class Finding(BaseModel):
    text: str
    category: str
    reason: str
    # The line the finding sits on. When set, only occurrences inside it
    # are masked, so a word that is sensitive in one place (an address)
    # stays visible elsewhere (the report title).
    context: str = ""


class DetectorResult(BaseModel):
    findings: list[Finding]


class ContextResult(BaseModel):
    additions: list[Finding]
    rejections: list[str]


class ReviewResult(BaseModel):
    findings: list[Finding]


class EntailmentScore(BaseModel):
    score: int = Field(ge=0, le=100)
