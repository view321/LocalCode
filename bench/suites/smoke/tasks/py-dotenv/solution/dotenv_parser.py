"""Tiny parser for .env files."""


def parse(text):
    """Parse the contents of a .env file into a dict of str -> str.

    Rules:
    - one KEY=VALUE assignment per line
    - blank lines and lines whose first non-space character is '#' are ignored
    - an optional leading 'export ' before KEY is ignored
    - whitespace around KEY and around VALUE is stripped
    - a VALUE wrapped in matching single or double quotes keeps its inner
      whitespace and '#' characters; the surrounding quotes are removed
    - in an unquoted VALUE, the first ' #' (space then hash) starts a trailing
      comment: everything from that space on is dropped, then the value is
      right-stripped
    - lines without '=' are ignored
    - a later assignment to the same KEY overrides an earlier one
    - 'KEY=' (nothing after '=') maps KEY to the empty string

    Returns the dict; an empty or comment-only input returns {}.
    """
    result = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        else:
            idx = value.find(" #")
            if idx != -1:
                value = value[:idx].rstrip()
        result[key] = value
    return result
