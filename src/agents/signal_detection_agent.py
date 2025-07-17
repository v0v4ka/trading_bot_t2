import logging
import os
from typing import Callable, Dict, List, Optional

import openai
import pandas as pd

from ..indicators.engine import IndicatorsEngine

logger = logging.getLogger("trading_bot.signal_detection")


class SignalDetectionAgent:
    """Detects entry signals using Bill Williams indicators and optional LLM confirmation."""

    def __init__(
        self, df: pd.DataFrame, llm_client: Optional[Callable[[str], float]] = None
    ):
        self.df = df
        self.engine = IndicatorsEngine(df)
        self.indicators = self.engine.calculate_all()
        self.llm_client = llm_client

    def _build_prompt(self, fractal: Dict, ao_value: float) -> str:
        template_path = os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "prompts",
            "signal_confirmation_prompt.txt",
        )
        template_path = os.path.normpath(template_path)
        try:
            with open(template_path, "r") as f:
                template = f.read()
        except FileNotFoundError:
            template = (
                "Fractal type: {fractal_type}\n"
                "AO value: {ao_value}\n"
                "Provide a confidence score between 0 and 1."
            )
        return template.format(fractal_type=fractal["type"], ao_value=f"{ao_value:.2f}")

    def _default_llm_client(self, prompt: str) -> float:
        try:
            client = openai.OpenAI()
            completion = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            )
            content = completion.choices[0].message.content
            return float(content.strip() if content else "0.5")
        except Exception as e:
            logger.error(f"LLM client error: {e}")
            return 0.5

    def detect_signals(self) -> List[Dict]:
        fractals = self.indicators["fractals"]
        ao = self.indicators["awesome_oscillator"]
        signals = []
        for fr in fractals:
            idx = self.df.index.get_loc(fr["timestamp"])
            ao_value = ao.iloc[idx]
            if pd.isna(ao_value):
                continue
            if fr["type"] == "up" and ao_value > 0:
                action = "buy"
            elif fr["type"] == "down" and ao_value < 0:
                action = "sell"
            else:
                continue
            base_confidence = 0.5
            llm_conf = 0.0
            if self.llm_client:
                prompt = self._build_prompt(fr, ao_value)
                llm_conf = self.llm_client(prompt)
            else:
                prompt = self._build_prompt(fr, ao_value)
                llm_conf = self._default_llm_client(prompt)
            confidence = min(1.0, base_confidence + 0.5 * llm_conf)
            signals.append(
                {
                    "timestamp": fr["timestamp"],
                    "type": action,
                    "confidence": confidence,
                    "details": fr,
                }
            )
        return signals
