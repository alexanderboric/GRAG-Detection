import time
import logging
from functools import wraps
from tqdm import tqdm
from scripts.config_loader import get_config

# Mapping von String-Loglevel zu logging-Level
levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
}

logger = logging.getLogger(__name__)  # Modul-Logger
_logging_initialized = False          # Flag, ob basicConfig schon gesetzt wurde

# internal tqdm progress instance (keeps bar at bottom)
_tqdm_progress = None


class TqdmLoggingHandler(logging.Handler):
    """Logging handler that routes messages through tqdm.write so logs
    appear above a tqdm progress bar instead of clobbering it.
    """
    def emit(self, record: logging.LogRecord) -> None:
        try:
            from tqdm import tqdm
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            # fallback to standard stream if tqdm not available
            print(self.format(record))


def _ensure_logging_configured():
    """Sorgt dafür, dass logging.basicConfig genau einmal sinnvoll gesetzt wird."""
    global _logging_initialized

    if _logging_initialized:
        return

    # Versuche, Config zu laden – falls noch nicht geladen, nimm Standardwerte
    try:
        cfg = get_config()
        verbosity = cfg.get("logging", {}).get("verbosity", "debug").lower()
        # print(f"Config verbosity: {verbosity}")
        conf_val = levels.get(verbosity, logging.DEBUG)
    except RuntimeError:
        # Config noch nicht geladen → Default Debug
        conf_val = logging.DEBUG

    # Configure file handler and a tqdm-aware stream handler so
    # logs are written above any tqdm progress bar.
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(funcName)s: %(message)s")

    fh = logging.FileHandler("app.log")
    fh.setLevel(conf_val)
    fh.setFormatter(fmt)

    try:
        # Use tqdm-aware handler
        th = TqdmLoggingHandler()
        th.setLevel(conf_val)
        th.setFormatter(fmt)
    except Exception:
        th = logging.StreamHandler()
        th.setLevel(conf_val)
        th.setFormatter(fmt)

    # Attach handlers to module logger
    logger.setLevel(conf_val)
    logger.addHandler(fh)
    logger.addHandler(th)
    _logging_initialized = True


# decorator to log the time taken by any given function
def log_time(stage_name: str):
    """Dekorator, um die Laufzeit einer Funktion zu messen und zu loggen."""
    import asyncio as _asyncio
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            _ensure_logging_configured()
            logger.info(f"[{stage_name}] Starting.")
            start = time.perf_counter()
            result = await func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"[{stage_name}] Done. Took {elapsed:.4f} seconds.")
            return result

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            _ensure_logging_configured()
            logger.info(f"[{stage_name}] Starting.")
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            logger.info(f"[{stage_name}] Done. Took {elapsed:.4f} seconds.")
            return result

        return async_wrapper if _asyncio.iscoroutinefunction(func) else sync_wrapper
    return decorator


# ------------------------------------------------------------
# Verbose-Print / Logging-Helfer
# ------------------------------------------------------------

def printv(message: str, level: str = "debug", **kwargs):
    """
    Loggt eine Nachricht entsprechend des in der Config gesetzten Loglevels.
    Fällt auf DEBUG zurück, wenn noch keine Config geladen ist.
    """
    _ensure_logging_configured()

    log_level = levels.get(level.lower(), logging.DEBUG)

    # Versuche, die aktuelle Verbosity aus der Config zu lesen
    try:
        cfg = get_config()
        conf_val = levels.get(
            cfg.get("logging", {}).get("verbosity", "debug").lower(),
            logging.DEBUG,
        )
    except RuntimeError:
        # Config noch nicht geladen → Debug als Default
        conf_val = logging.DEBUG

    # Nur loggen, wenn Nachricht-Level >= konfigurierter Level
    if log_level >= conf_val:
        logger.log(log_level, message, stacklevel=2, **kwargs)



# show a persistent bottom progress bar using tqdm.
# logs are routed above the bar using TqdmLoggingHandler.
def progress_bar(current: int, total: int, bar_length: int = 20):
    

    global _tqdm_progress

    # create tqdm instance if needed or if total changed
    if _tqdm_progress is None or getattr(_tqdm_progress, 'total', None) != total:
        if _tqdm_progress is not None:
            try:
                _tqdm_progress.close()
            except Exception:
                pass
        # use a filled bar format, let tqdm render the fill
        bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        _tqdm_progress = tqdm(total=total, leave=True, ncols=80, position=0, bar_format=bar_fmt, ascii=False)

    # update bar (tqdm expects increments; set n directly and refresh)
    try:
        _tqdm_progress.n = int(current)
        _tqdm_progress.refresh()
        if current >= total:
            _tqdm_progress.close()
            _tqdm_progress = None
    except Exception:
        pass


def make_progress_ticker(total: int):
    # returns a no-arg callback that advances progress_bar by one step each call. Guarded by a
    # lock since detector.detect_batch() now calls this concurrently from a ThreadPoolExecutor
    # for network-bound detectors - the bare `i += 1` read-modify-write isn't atomic.
    import threading
    i = 0
    lock = threading.Lock()
    def tick():
        nonlocal i
        with lock:
            i += 1
            progress_bar(current=i, total=total)
    return tick


def retry_api_call(fn, *args, retries: int = 3, base_delay: float = 2.0, **kwargs):
    """Calls fn(*args, **kwargs), retrying with exponential backoff on transient API errors
    (connection drop, timeout, 5xx) - re-raises unchanged on the final attempt or for any other
    exception type, so real bugs still surface immediately rather than being masked. Use this at
    every network-calling site (embeddings, chat completions, local_search) instead of duplicating
    retry logic per call site - a single transient blip during a multi-hour unattended run
    shouldn't abort the whole run."""
    from openai import APIConnectionError, APIError, APITimeoutError

    for attempt in range(1, retries + 1):
        try:
            return fn(*args, **kwargs)
        except (APIConnectionError, APITimeoutError, APIError, TimeoutError) as e:
            if attempt == retries:
                raise
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning(f"API call failed (attempt {attempt}/{retries}): {e}. Retrying in {wait:.1f}s...")
            time.sleep(wait)
