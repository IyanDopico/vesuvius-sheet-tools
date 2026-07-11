"""List available data for a Vesuvius Challenge sample in the public S3 bucket.

Usage:
    python scripts/list_scroll_data.py PHerc0332
    python scripts/list_scroll_data.py PHercParis4 segments/
"""

import sys
import xml.etree.ElementTree as ET
from urllib.request import urlopen

BUCKET_URL = "https://vesuvius-challenge-open-data.s3.amazonaws.com"
NS = "{http://s3.amazonaws.com/doc/2006-03-01/}"


def list_prefix(prefix: str) -> list[str]:
    """Return immediate child prefixes/keys under `prefix` (delimiter '/')."""
    results: list[str] = []
    token = ""
    while True:
        url = f"{BUCKET_URL}/?list-type=2&delimiter=/&prefix={prefix}{token}"
        with urlopen(url) as resp:
            root = ET.fromstring(resp.read())
        results += [e.text for e in root.iter(f"{NS}Prefix") if e.text and e.text != prefix]
        results += [e.text for e in root.iter(f"{NS}Key") if e.text]
        next_token = root.find(f"{NS}NextContinuationToken")
        if next_token is None:
            break
        token = f"&continuation-token={next_token.text}"
    return results


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    sample = sys.argv[1].rstrip("/")
    sub = sys.argv[2] if len(sys.argv) > 2 else ""
    prefix = f"{sample}/{sub}"
    entries = list_prefix(prefix)
    if not entries:
        print(f"No entries under {prefix!r}")
        return
    print(f"{len(entries)} entries under {prefix!r}:")
    for e in sorted(set(entries)):
        print(f"  {e}")


if __name__ == "__main__":
    main()
