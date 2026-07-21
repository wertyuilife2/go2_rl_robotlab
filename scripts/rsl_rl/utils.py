"""Utilities shared by the RSL-RL command-line scripts."""

import os
import re
import sys


class Logger:
    """Mirror stdout to a file while removing ANSI escape sequences."""

    def __init__(self, filename):
        """Initialize the terminal and file outputs.

        Args:
            filename: Destination log file path.
        """
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, "w", encoding="utf-8")
        self.ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")

    def write(self, message):
        """Write a message to the terminal and cleaned log file.

        Args:
            message: Text to write.
        """
        clean_message = self.ansi_escape.sub("", message)
        self.terminal.write(message)
        self.log.write(clean_message)
        self.log.flush()

    def flush(self):
        """Flush terminal and log file buffers."""
        self.terminal.flush()
        self.log.flush()
