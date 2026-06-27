import json
import logging
import os
from datetime import datetime

from config import LOG_DIR

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, f"app_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(module)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# Dedicated structured log for model calls (the model-selection dashboard reads this).
MODEL_EVENT_LOG = os.path.join(LOG_DIR, "model_events.jsonl")


def log_model_event(event) -> None:
    """Append a ModelEvent (or dict) as a JSON line and echo to the main log.

    Accepts either a `models.ModelEvent` instance or a plain dict so callers
    don't have to import the schema just to log.
    """
    payload = event.model_dump() if hasattr(event, "model_dump") else dict(event)
    try:
        with open(MODEL_EVENT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError as exc:  # logging must never crash the run
        logger.warning("Failed to persist model event: %s", exc)
    logger.info(
        "model_event agent=%s model=%s latency_ms=%s schema_valid=%s escalated=%s error=%s",
        payload.get("agent"),
        payload.get("model"),
        payload.get("latency_ms"),
        payload.get("schema_valid"),
        payload.get("escalated"),
        payload.get("error"),
    )
