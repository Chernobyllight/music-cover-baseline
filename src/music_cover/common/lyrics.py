import re

def clean_lyrics(text):
    text = text.lstrip("\ufeff")
    rows, timed = [], []
    timestamp = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d+))?\]")
    for index, line in enumerate(text.splitlines()):
        tags = list(timestamp.finditer(line))
        line = timestamp.sub("", line).strip()
        if re.match(r"^\[(?:ar|ti|al|by|offset|length):", line, re.I):
            continue
        if re.match(r"^(?:作词|作曲|编曲|制作人|混音|母带|词|曲)\s*[:：]", line):
            continue
        rows.append(line)
        if line:
            for tag in tags:
                fraction = float("0." + tag[3]) if tag[3] else 0.0
                timed.append((int(tag[1]) * 60 + int(tag[2]) + fraction, index, line))
    if timed:
        rows = [line for _, _, line in sorted(timed)]
    cleaned = "\n".join(rows).strip()
    if not cleaned or not re.search(r"[\w\u4e00-\u9fff]", cleaned):
        raise ValueError("Lyrics are empty after removing metadata")
    if not re.search(r"\[(?:Verse|Chorus|Intro|Bridge|Outro|Pre-Chorus|Interlude)\]", cleaned, re.I):
        cleaned = "[Verse]\n" + cleaned
    return cleaned + "\n"
