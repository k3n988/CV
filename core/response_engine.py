"""
response_engine.py

Rules-based mapping from aggregated signals to a response tier and
message, per the concept paper's three-tier design:

  Tier 1 - mild/neutral cues: no action, or a light friendly check-in.
  Tier 2 - visible sadness/frustration: scripted comfort message,
           breathing prompt, or journaling prompt.
  Tier 3 - cues consistent with severe/prolonged distress: stop trying
           to "handle" it, surface real support resources.

Thresholds and message content are kept as plain data (not hardcoded
branching logic) so they can be tuned without touching the rest of
the app.
"""

import random

# Tunable thresholds -- adjust these after real-world testing with your
# aggregation window.
TIER2_EXPRESSION_RATIO_THRESHOLD = 0.6   # e.g. sad/angry dominant in 60%+ of window
TIER3_EXPRESSION_RATIO_THRESHOLD = 0.8   # sustained, high-ratio negative signal
TIER3_MIN_SAMPLE_COUNT = 20              # require enough samples before Tier 3 fires
                                          # (avoids false positives from a short window)

NEGATIVE_EXPRESSIONS = {"sad", "angry", "fear", "disgust"}

TIER1_MESSAGES = [
    "Hey, just checking in -- how's it going?",
    "All good here whenever you're ready.",
]

TIER2_MESSAGES = [
    "Looks like today might be a bit heavy. Want to try a quick breathing exercise?",
    "It's okay to feel this way. Would writing a short note about it help?",
    "Take a moment if you need it -- no rush.",
]

# Tier 3 always points to real, verifiable human support channels.
# NOTE: verify this number is current before using in any real deployment.
TIER3_RESOURCES = {
    "message": (
        "It looks like things might be really difficult right now. "
        "You deserve real support, not just an app message."
    ),
    "resources": [
        {
            "name": "National Center for Mental Health (NCMH) Crisis Line (Philippines)",
            "contact": "1553 (toll-free landline) / 0966-351-4518 or 0908-639-2672 (mobile)",
        },
        {
            "name": "In case of immediate danger",
            "contact": "Contact local emergency services or go to the nearest hospital.",
        },
    ],
}


class ResponseEngine:
    def evaluate(self, aggregate: dict):
        """
        Takes the aggregate dict from SignalAggregator.get_aggregate()
        and returns a dict describing the response tier and content:
        {
            "tier": 1 | 2 | 3,
            "message": str,
            "resources": list | None,
        }
        """
        if aggregate is None:
            return {"tier": 1, "message": random.choice(TIER1_MESSAGES), "resources": None}

        dominant = aggregate["dominant_expression"]
        ratio = aggregate["expression_ratio"]
        head_down_ratio = aggregate["head_down_ratio"]
        still_ratio = aggregate["still_ratio"]
        sample_count = aggregate["sample_count"]

        is_negative = dominant in NEGATIVE_EXPRESSIONS
        posture_supports_distress = head_down_ratio > 0.6 or still_ratio > 0.6

        # Tier 3: sustained negative expression + supporting posture cues,
        # with enough samples to be confident this isn't a fluke.
        if (
            is_negative
            and ratio >= TIER3_EXPRESSION_RATIO_THRESHOLD
            and posture_supports_distress
            and sample_count >= TIER3_MIN_SAMPLE_COUNT
        ):
            return {
                "tier": 3,
                "message": TIER3_RESOURCES["message"],
                "resources": TIER3_RESOURCES["resources"],
            }

        # Tier 2: noticeable but not sustained/severe negative signal.
        if is_negative and ratio >= TIER2_EXPRESSION_RATIO_THRESHOLD:
            return {"tier": 2, "message": random.choice(TIER2_MESSAGES), "resources": None}

        # Tier 1: default, mild/neutral.
        return {"tier": 1, "message": random.choice(TIER1_MESSAGES), "resources": None}
