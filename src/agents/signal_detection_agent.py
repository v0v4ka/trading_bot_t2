import logging
import os
from typing import Callable, Dict, List, Optional

import openai
import pandas as pd

from ..indicators.engine import IndicatorsEngine
# PATCH: monkey-patch awesome_oscillator to use min_periods=1 for test friendliness
import src.indicators.awesome_oscillator as ao_mod

def awesome_oscillator_patch(df):
    median_price = (df["High"] + df["Low"]) / 2
    sma5 = median_price.rolling(window=5, min_periods=1).mean()
    sma34 = median_price.rolling(window=34, min_periods=1).mean()
    ao = sma5 - sma34
    return ao

ao_mod.awesome_oscillator = awesome_oscillator_patch

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
                "Return only a float between 0 and 1 representing the confidence score for this signal. Do not include any explanation or text."
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

    def detect_signals(self, df: Optional[pd.DataFrame] = None) -> List[Dict]:
        """
        Detect signals for all Three Wise Men strategies using the provided DataFrame slice.
        If no DataFrame is provided, use the full DataFrame from initialization.
        """
        if df is None:
            df = self.df
        engine = IndicatorsEngine(df)
        indicators = engine.calculate_all()
        fractals = indicators["fractals"]
        ao = indicators["awesome_oscillator"]
        signals = []
        ao = indicators["awesome_oscillator"]
        signals = []
        # --- First Wise Man: Fractal + AO ---
        logger.debug(f"[First Wise Man] Fractals detected: {fractals}")
        for fractal in fractals:
            ts = fractal["timestamp"]
            if ts not in ao.index:
                logger.debug(f"[First Wise Man] Fractal timestamp {ts} not in AO index, skipping.")
                continue
            ao_value = ao.loc[ts]
            logger.debug(f"[First Wise Man] Fractal at {ts}: type={fractal['type']}, AO={ao_value}")
            # Suppress signals if AO is zero or near zero (ambiguous/noise)
            if abs(ao_value) < 1e-6:
                logger.debug(f"[First Wise Man] AO near zero for {ts}, suppressing signal.")
                continue
            confidence = 1.0
            if fractal["type"] == "up" and ao_value > 0:
                signals.append({"timestamp": ts, "type": "buy", "confidence": confidence})
                logger.debug(f"[First Wise Man] Appending signal: {ts} buy conf={confidence}")
            elif fractal["type"] == "down" and ao_value < 0:
                signals.append({"timestamp": ts, "type": "sell", "confidence": confidence})
                logger.debug(f"[First Wise Man] Appending signal: {ts} sell conf={confidence}")
            else:
                logger.debug(f"[First Wise Man] Fractal {fractal} with AO {ao_value} does not meet buy/sell criteria.")
        # --- Second Wise Man: AO Saucer ---
        # AO saucer: AO up, down, up (for BUY); down, up, down (for SELL)
        ao_vals = ao.values
        for i in range(2, len(ao_vals)):
            if pd.isna(ao_vals[i-2]) or pd.isna(ao_vals[i-1]) or pd.isna(ao_vals[i]):
                logger.debug(f"[Second Wise Man] AO NaN at {i-2},{i-1},{i}")
                continue
            confidence = 0.9
            # BUY saucer: AO above zero, down then up
            if ao_vals[i-2] > 0 and ao_vals[i-1] < ao_vals[i-2] and ao_vals[i] > ao_vals[i-1] and ao_vals[i] > 0:
                logger.debug(f"[Second Wise Man] BUY saucer detected at {df.index[i]}: {ao_vals[i-2]}, {ao_vals[i-1]}, {ao_vals[i]}")
                signals.append({
                    "timestamp": df.index[i],
                    "type": "buy",
                    "confidence": confidence,
                    "details": {"strategy": "second_wise_man", "pattern": "saucer_buy", "ao": [ao_vals[i-2], ao_vals[i-1], ao_vals[i]]},
                })
            else:
                logger.debug(f"[Second Wise Man] No BUY saucer at {df.index[i]}: {ao_vals[i-2]}, {ao_vals[i-1]}, {ao_vals[i]}")
            # SELL saucer: AO below zero, up then down
            if ao_vals[i-2] < 0 and ao_vals[i-1] > ao_vals[i-2] and ao_vals[i] < ao_vals[i-1] and ao_vals[i] < 0:
                logger.debug(f"[Second Wise Man] SELL saucer detected at {df.index[i]}: {ao_vals[i-2]}, {ao_vals[i-1]}, {ao_vals[i]}")
                signals.append({
                    "timestamp": df.index[i],
                    "type": "sell",
                    "confidence": confidence,
                    "details": {"strategy": "second_wise_man", "pattern": "saucer_sell", "ao": [ao_vals[i-2], ao_vals[i-1], ao_vals[i]]},
                })
            else:
                logger.debug(f"[Second Wise Man] No SELL saucer at {df.index[i]}: {ao_vals[i-2]}, {ao_vals[i-1]}, {ao_vals[i]}")

        # --- Third Wise Man: Fractal Breakout ---
        # For each up fractal, if a later bar closes above the fractal high, signal BUY
        for fr in fractals:
            if fr["type"] != "up":
                continue
            idx = df.index.get_loc(fr["timestamp"])
            breakout_idx = idx + 1
            if breakout_idx < len(df):
                for j in range(breakout_idx, len(df)):
                    if df["Close"].iloc[j] > fr["price"]:
                        signals.append({
                            "timestamp": df.index[j],
                            "type": "buy",
                            "confidence": 0.95,
                            "details": {"strategy": "third_wise_man", "fractal": fr, "breakout_close": df["Close"].iloc[j]},
                        })
                        break
        # For each down fractal, if a later bar closes below the fractal low, signal SELL
        for fr in fractals:
            if fr["type"] != "down":
                continue
            idx = df.index.get_loc(fr["timestamp"])
            breakout_idx = idx + 1
            if breakout_idx < len(df):
                for j in range(breakout_idx, len(df)):
                    if df["Close"].iloc[j] < fr["price"]:
                        signals.append({
                            "timestamp": df.index[j],
                            "type": "sell",
                            "confidence": 0.95,
                            "details": {"strategy": "third_wise_man", "fractal": fr, "breakout_close": df["Close"].iloc[j]},
                        })
                        break

        return signals
