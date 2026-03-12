"""
Prompt template manager.

Loads prompt templates from configs/prompts.yaml and provides
a clean interface for rendering them with runtime variables.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.llm import Message

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def _load_prompts(prompts_path: str) -> dict[str, Any]:
    """Load and cache the prompts YAML file."""
    path = Path(prompts_path)
    if not path.exists():
        logger.warning("prompts.file_not_found", path=str(path))
        return {}
    with path.open() as fh:
        return yaml.safe_load(fh) or {}


class PromptManager:
    """
    Manages prompt templates for all LLM calls.

    Templates are stored in configs/prompts.yaml and use Python str.format()
    style placeholders: {variable_name}.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._prompts = _load_prompts(str(self._settings.prompts_path))

    def get_messages(
        self,
        prompt_key: str,
        variables: dict[str, Any] | None = None,
    ) -> list[Message]:
        """
        Build a list of chat messages for the given prompt template.

        Args:
            prompt_key: Key in prompts.yaml (e.g. "intent_extraction")
            variables: Dictionary of {placeholder: value} for template rendering

        Returns:
            List of Message objects ready to send to an LLM provider.
        """
        variables = variables or {}
        template = self._prompts.get(prompt_key, {})

        if not template:
            logger.warning("prompts.key_not_found", key=prompt_key)
            return self._fallback_messages(prompt_key, variables)

        messages = []

        system_raw = template.get("system", "")
        if system_raw:
            system_content = self._render(system_raw, variables)
            messages.append(Message(role="system", content=system_content))

        user_raw = template.get("user", "")
        if user_raw:
            user_content = self._render(user_raw, variables)
            messages.append(Message(role="user", content=user_content))

        return messages

    def _render(self, template: str, variables: dict[str, Any]) -> str:
        """Render a template string with the given variables."""
        try:
            return template.format(**{k: str(v) for k, v in variables.items()})
        except KeyError as exc:
            logger.warning("prompts.render_missing_key", key=str(exc), template=template[:80])
            # Return template with unfilled placeholders replaced by empty string
            import re
            return re.sub(r"\{[^}]+\}", "", template)

    def _fallback_messages(
        self, prompt_key: str, variables: dict[str, Any]
    ) -> list[Message]:
        """Minimal fallback messages when the template is missing."""
        context = str(variables)[:500]
        return [
            Message(
                role="system",
                content=f"You are TripGenie, a helpful travel planning assistant. Task: {prompt_key}.",
            ),
            Message(
                role="user",
                content=f"Please help with the following: {context}",
            ),
        ]
