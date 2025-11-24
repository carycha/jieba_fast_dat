import re


def _parse_dict_line(line: str) -> tuple[str, int, str | None]:
    """
    Parses a line from a dictionary file.
    A line can be in one of these formats:
    1. word freq tag
    2. word freq
    3. word tag (freq defaults to 1)
    4. word (freq defaults to 1, tag is None)
    """
    line = line.strip()
    if not line:
        raise ValueError("Empty line cannot be parsed.")

    parts = re.split(r"\s+", line)
    word = parts[0]
    freq = 1
    tag = None

    if len(parts) == 2:
        # Could be "word freq" or "word tag"
        try:
            freq = int(parts[1])
        except ValueError:
            # It's "word tag"
            tag = parts[1]
    elif len(parts) >= 3:
        # Must be "word freq tag"
        try:
            freq = int(parts[1])
            tag = parts[2]
        except ValueError:
            # Handle cases like "word tag tag2" where freq is missing
            # For simplicity, we'll assume the second part is freq if it's a number
            # Otherwise, treat it as "word tag" with default freq=1
            tag = parts[1]  # Treat second part as tag if not freq
            if len(parts) > 2:
                tag = (
                    parts[1] + " " + " ".join(parts[2:])
                )  # Concatenate remaining as tag

    return word, freq, tag
