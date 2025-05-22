from datetime import datetime

def now() -> str:
    return datetime.now().strftime("%Y‑%m‑%d %H:%M:%S")


def indent(text: str, pad: int = 2) -> str:
    prefix = " " * pad
    return "\n".join(prefix + ln for ln in text.splitlines())