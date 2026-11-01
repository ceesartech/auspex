"""Text decoding for the football-data loaders, and the mojibake test.

football-data.co.uk's CSVs come in two encodings (all 346 files of the 16
European codes 2007-08..2026-27 and the 16 /new/ country files, fetched
2026-11-01): the per-season files up to 2023-24 are ASCII, two of them with a
stray Windows-1252 no-break space (0xA0) in an odds column (2018-19 I2, SC0);
the files from 2024-25 on, three older seasons football-data re-published
(2018-19 T1, 2021-22 E0, 2022-23 F2) and every /new/ country file are UTF-8
with a byte-order mark. Until 2026-11-01 both loaders decoded every file as
latin-1, which never fails and so never told anyone:

  * the UTF-8 'Preußen Münster' (2. Bundesliga 2024-25 and 2025-26) was
    stored as 'PreuÃ\\x9fen MÃ¼nster', in raw_football_data and as a team;
  * the BOM decoded to 'ï»¿' at the head of the first column's name, so the
    'Div' column of every BOM file was dropped (it read as 'ï»¿Div'), and with
    it the only check that a file holds the division it was fetched for. On
    2026-08-06 football-data served the Scottish Championship and League One
    openers of 2026-08-01 at the 2026-27 SP1 and SP2 URLs; ten Scottish
    fixtures and twenty clubs were filed under La Liga / La Liga 2.

decode_csv_bytes reads UTF-8 first (the BOM stripped), then Windows-1252,
strictly both times: bytes that are neither raise UndecodableText instead of
becoming text nobody wrote. Trying UTF-8 first is safe: Windows-1252 text
reads as UTF-8 only when its non-ASCII characters happen to pair into UTF-8
sequences ('Ã©'), which real names never do — a lone accented letter
(0x80-0xFF followed by ASCII) is not a UTF-8 sequence.

looks_mojibake is the test for text that was decoded wrongly upstream: UTF-8
bytes read as latin-1 or Windows-1252 leave a C1 control character
(U+0080-U+009F, never in a name) or a lead character Â-ô (0xC2-0xF4)
followed by what a continuation byte 0x80-0xBF reads as; a BOM or U+FFFD
left in the text counts too. MOJIBAKE_PATTERN is the same test for Postgres
(`~`), for the repair tool and the after-state sweeps. repair_mojibake undoes
a latin-1 or Windows-1252 misreading of UTF-8 (applied up to twice, for text
misread twice), or returns None when the result would still look wrong.
"""

from __future__ import annotations

import re

# Windows-1252's readings of the bytes 0x80-0x9F (the ones it defines): the
# characters a UTF-8 continuation byte in that range turns into.
_CP1252_C1 = (
    "\\u20ac\\u201a\\u0192\\u201e\\u2026\\u2020\\u2021\\u02c6\\u2030\\u0160\\u2039\\u0152\\u017d"
    "\\u2018\\u2019\\u201c\\u201d\\u2022\\u2013\\u2014\\u02dc\\u2122\\u0161\\u203a\\u0153\\u017e\\u0178"
)
# One pattern for Python's re and for Postgres's `~` (both read \\uXXXX).
MOJIBAKE_PATTERN = f"[\\u0080-\\u009f]|[\\u00c2-\\u00f4][\\u00a0-\\u00bf{_CP1252_C1}]|\\ufeff|\\ufffd"
_MOJIBAKE = re.compile(MOJIBAKE_PATTERN)


class UndecodableText(ValueError):
    """Bytes that are neither UTF-8 nor Windows-1252."""


def decode_csv_bytes(raw: bytes, source: str) -> tuple[str, str]:
    """(text, encoding) of a downloaded CSV: UTF-8 (a leading BOM dropped),
    else Windows-1252, each strict. Raises UndecodableText naming `source`
    and the offending byte when neither reads it."""
    try:
        return raw.decode("utf-8-sig"), "utf-8"
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252"), "cp1252"
    except UnicodeDecodeError as exc:
        raise UndecodableText(
            f"{source}: neither UTF-8 nor Windows-1252 (byte 0x{raw[exc.start]:02x} at offset {exc.start}) — "
            "not loaded; find out what encoding the file is in before reading it"
        ) from None


def looks_mojibake(text: str | None) -> bool:
    """Whether `text` holds the marks of a wrong decode (see the module docstring)."""
    return text is not None and _MOJIBAKE.search(text) is not None


def _undo_once(text: str) -> str | None:
    for codec in ("latin-1", "cp1252"):
        try:
            return text.encode(codec).decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return None


def repair_mojibake(text: str) -> str | None:
    """`text` decoded right, when it is UTF-8 that was read as latin-1 or
    Windows-1252 (once or twice); None when it does not look wrong or no such
    reading leaves clean text. 'PreuÃ\\x9fen MÃ¼nster' -> 'Preußen Münster'."""
    if not looks_mojibake(text):
        return None
    current = text
    for _ in range(2):
        undone = _undo_once(current)
        if undone is None or undone == current:
            return None
        if not looks_mojibake(undone):
            return undone
        current = undone
    return None
