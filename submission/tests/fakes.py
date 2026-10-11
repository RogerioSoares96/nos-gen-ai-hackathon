import threading


class FakeLLMClient:
    """LLMClient double driven by a responder(prompt, schema) function.

    Routing on the prompt (not on call order) keeps tests deterministic
    when the orchestrator calls agents in parallel. A str reply is parsed
    with the schema, exactly as GeminiClient does.
    """

    def __init__(self, responder):
        self.responder = responder
        self.prompts = []
        self._lock = threading.Lock()

    def generate_json(self, prompt, schema):
        self._record(prompt)
        reply = self.responder(prompt, schema)
        if isinstance(reply, str):
            return schema.model_validate_json(reply)
        return reply

    def generate_text(self, prompt):
        self._record(prompt)
        return self.responder(prompt, None)

    def _record(self, prompt):
        with self._lock:
            self.prompts.append(prompt)
