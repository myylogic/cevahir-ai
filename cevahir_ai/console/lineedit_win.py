"""A small line editor for Windows consoles without readline: history, cursor keys and Tab.

Python on Windows has no ``readline``; ``input()`` gives the console's own editing and history
but no completion. This editor reads keys with ``msvcrt.getwch`` and redraws the line with plain
carriage returns (no escape sequences), so it works in ``cmd.exe``, in the PowerShell terminal and
in Windows Terminal. Tab cycles through the completions, as PowerShell does.

The key source and the writer are parameters, so the editor is tested with scripted keys on any
platform. It has not been run on a real Windows console yet: ``CEVAHIR_LINE_EDITOR=plain`` switches
back to ``input()``.
"""

SPECIAL = {"H": "up", "P": "down", "K": "left", "M": "right", "G": "home", "O": "end", "S": "delete"}


def word_start(text, cursor):
    """Index where the word under the cursor starts (the unit that Tab replaces)."""
    index = cursor
    while index > 0 and text[index - 1] not in " \t|;(){},":
        index -= 1
    return index


class LineEditor:
    def __init__(self, getwch, write, complete=None, history=None):
        self.getwch, self.write, self.complete = getwch, write, complete
        self.history = history if history is not None else []

    def read_line(self, prompt):
        """Read one line. Raises ``KeyboardInterrupt`` on Ctrl+C and ``EOFError`` on Ctrl+Z at an empty line."""
        text, cursor, shown = "", 0, 0
        position = len(self.history)          # index into history; len(history) is the line being typed
        pending = ""
        cycle = None                          # (start of the word, candidates, index) while Tab is being pressed

        def redraw():
            nonlocal shown
            self.write("\r" + prompt + text + " " * max(shown - len(text), 0) + "\r" + prompt + text[:cursor])
            shown = len(text)

        self.write(prompt)
        while True:
            key = self.getwch()
            if key in ("\x00", "\xe0"):
                key = SPECIAL.get(self.getwch(), "")
            if key != "\t":
                cycle = None
            if key in ("\r", "\n"):
                self.write("\r\n")
                if text.strip() and (not self.history or self.history[-1] != text):
                    self.history.append(text)
                return text
            if key == "\x03":
                self.write("^C\r\n")
                raise KeyboardInterrupt
            if key == "\x1a" and not text:
                self.write("\r\n")
                raise EOFError
            if key == "\x1b":
                text, cursor = "", 0
            elif key == "\x08":
                if cursor:
                    text, cursor = text[:cursor - 1] + text[cursor:], cursor - 1
            elif key == "delete":
                text = text[:cursor] + text[cursor + 1:]
            elif key == "left":
                cursor = max(cursor - 1, 0)
            elif key == "right":
                cursor = min(cursor + 1, len(text))
            elif key == "home":
                cursor = 0
            elif key == "end":
                cursor = len(text)
            elif key in ("up", "down"):
                if position == len(self.history):
                    pending = text
                position = max(position - 1, 0) if key == "up" else min(position + 1, len(self.history))
                text = self.history[position] if position < len(self.history) else pending
                cursor = len(text)
            elif key == "\t":
                if self.complete is not None:
                    if cycle is None:
                        start = word_start(text, cursor)
                        candidates = [item for item in self.complete(text, cursor) if item]
                        cycle = (start, candidates, -1, text[cursor:]) if candidates else None
                    if cycle is not None:
                        start, candidates, index, tail = cycle
                        index = (index + 1) % len(candidates)
                        cycle = (start, candidates, index, tail)
                        text = text[:start] + candidates[index] + tail
                        cursor = start + len(candidates[index])
            elif key and key >= " ":
                text, cursor = text[:cursor] + key + text[cursor:], cursor + 1
            redraw()
