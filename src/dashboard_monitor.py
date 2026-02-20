import logging
import threading
from collections import deque
from datetime import datetime, timezone

from .config import DASHBOARD_LOG_BUFFER_SIZE

# 保留最近 N 条完整日志（默认 1000）
_LOG_BUFFER_SIZE = max(1, DASHBOARD_LOG_BUFFER_SIZE)
_log_buffer = deque(maxlen=_LOG_BUFFER_SIZE)
_log_lock = threading.Lock()


class InMemoryLogHandler(logging.Handler):
    """将完整日志写入内存环形缓冲区。"""

    def emit(self, record: logging.LogRecord):
        try:
            formatted = self.format(record)
            entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "formatted": formatted,
            }
            with _log_lock:
                _log_buffer.append(entry)
        except Exception:
            # 日志 handler 不能影响主流程
            pass


def init_inmemory_log_handler():
    """初始化全局日志采集 handler，仅注册一次。"""
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, InMemoryLogHandler):
            return

    memory_handler = InMemoryLogHandler()
    memory_handler.setLevel(logging.NOTSET)
    memory_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    root_logger.addHandler(memory_handler)


def get_recent_logs(limit: int = 200):
    limit = max(1, min(limit, _LOG_BUFFER_SIZE))
    with _log_lock:
        items = list(_log_buffer)[-limit:]
    return items


def get_log_overview():
    with _log_lock:
        total = len(_log_buffer)
    return {
        "buffer_size": _LOG_BUFFER_SIZE,
        "buffered": total,
    }
