"""
Windows logging compatibility helpers.

On Windows, Python defaults to the GBK code page while Ray log messages contain
emoji characters (such as warning and check-mark symbols) that GBK cannot
encode. This causes noisy "--- Logging error ---" output from logging handlers.

Root cause: StreamHandler.emit() catches exceptions internally and calls
self.handleError(), so UnicodeEncodeError does not propagate. The correct
interception point is Handler.handleError, not StreamHandler.emit.
"""
import sys
import logging

_PATCHED = False


def patch_windows_unicode_logging() -> None:
    """
    Fix two logging issues with Unicode characters under the Windows GBK codec.

    **Problem 1 - UnicodeEncodeError**
    Ray log messages contain emoji that GBK cannot encode. StreamHandler.emit()
    catches the write error and calls self.handleError(), producing
    "--- Logging error ---" output. The message is still emitted by another
    available handler, such as Ray's file or console handler. The fix detects
    UnicodeEncodeError in handleError and returns without printing an error.

    **Problem 2 - StopIteration leakage**
    When the Ray streaming executor stops, StopIteration from a background
    thread can leak into handleError through
    shutdown() -> logger.info() -> handler.emit(), producing a harmless but
    confusing "--- Logging error ---". The fix detects StopIteration in
    handleError and returns without printing an error.

    This function is a no-op outside Windows and runs only once (idempotent).
    """
    global _PATCHED
    if sys.platform != 'win32' or _PATCHED:
        return
    _PATCHED = True

    _orig_handle_error = logging.Handler.handleError

    def _safe_handle_error(self, record):
        _, exc, _ = sys.exc_info()
        if isinstance(exc, (UnicodeEncodeError, StopIteration)):
            # UnicodeEncodeError: Ray emoji output failed on a GBK stream; another
            # handler already emitted the message.
            # StopIteration: normal termination signal from the Ray streaming
            # executor, not a real error.
            return
        _orig_handle_error(self, record)

    logging.Handler.handleError = _safe_handle_error
