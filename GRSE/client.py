"""Actor client that actually forwards sampling temperature to the provider."""

from threading import Lock

from maskills.llm.client import LLMClient, _create_with_retry


class SamplingLLMClient(LLMClient):
    def __init__(self, config, temperature=0.7):
        super().__init__(config)
        self.temperature = temperature
        self._usage_lock = Lock()
        self.usage = {"input": 0, "output": 0}

    def chat(self, system_prompt, user_input, max_tokens=None):
        return self.chat_with_usage(system_prompt, user_input, max_tokens)[0]

    def chat_messages_with_usage(self, messages, max_tokens=None):
        params = {"model": self.model, "messages": messages}
        limit_key = ("max_completion_tokens" if any(
            name in self.model.lower() for name in ("o1", "o3", "gpt-5")
        ) else "max_tokens")
        params[limit_key] = max_tokens or self.config.max_tokens
        if self.temperature is not None:
            params["temperature"] = self.temperature
        response = _create_with_retry(self.raw_client.chat.completions.create, **params)
        usage = getattr(response, "usage", None)
        tokens = {"input": getattr(usage, "prompt_tokens", 0) or 0,
                  "output": getattr(usage, "completion_tokens", 0) or 0}
        with self._usage_lock:
            for key in tokens:
                self.usage[key] += tokens[key]
        return response.choices[0].message.content.strip(), tokens
