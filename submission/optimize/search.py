from submission.anonymizer.llm import LLMClient


def search(seed_prompt: str, train: list[tuple[str, str, str]],
           held_out: list[tuple[str, str, str]], client: LLMClient,
           n: int = 4, rounds: int = 3) -> list[tuple[str, float]]:
    """Search for a better prompt.txt (APE/OPRO-style, spec §11).

    `train` and `held_out` are evaluate.score.load_dataset entries
    (stem, text, masked). Each round runs every candidate single-shot on
    the train documents; its fitness is the score_masked F1 of that
    output, and output whose structure doesn't match scores 0. Gemini then
    writes `n` new candidates from the best ones and their errors. Gemini
    never judges its own output: fitness is always the ground truth.

    Returns (prompt, held-out F1) pairs, best first. The real PDF is never
    used here, because it is the test.
    """
    raise NotImplementedError
