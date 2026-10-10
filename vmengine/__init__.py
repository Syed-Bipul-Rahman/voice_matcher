"""Voice Match engine: the pipeline shared by the command-line tool (voice_match.py) and the desktop app.

  audio     reading, writing, measuring, splitting and polishing audio
  models    where the models live, one-time setup, offline loading
  pipeline  Session: analyze -> render phrases (fast or best-of) -> export
  server    long-running JSON-lines server used by the desktop app
"""

__version__ = "1.1.0"


class EngineError(Exception):
    """An error the user can act on. `code` is stable so the app can react to it."""

    def __init__(self, message, code="error"):
        super().__init__(message)
        self.code = code


class Cancelled(Exception):
    pass
