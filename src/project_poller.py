# 这是新文件 catfishcli/src/project_poller.py 的全部内容

import threading
from .config import GEMINI_PROJECT_IDS
from .auth import get_user_project_id # 导入原始的函数作为备用方案

# --- 状态管理 ---
# 使用一个简单的计数器来追踪下一个要使用的项目ID的索引
_project_index = 0
# 使用线程锁来确保在高并发情况下计数器不会出错
_lock = threading.Lock()

def get_next_project_id(credentials) -> str:
    """
    从项目ID列表中轮询获取下一个项目ID。
    如果列表为空，则回退到原始的单项目ID获取方式。
    """
    # 检查项目ID列表是否为空
    if not GEMINI_PROJECT_IDS:
        # 如果用户没有配置轮询列表，则使用原始的、从凭据中获取项目ID的方法
        return get_user_project_id(credentials)

    # 如果列表不为空，则执行轮询逻辑
    with _lock:
        global _project_index
        
        # 从列表中获取当前索引对应的项目ID
        project_id = GEMINI_PROJECT_IDS[_project_index]
        
        # 更新索引，让它指向下一个项目ID，如果到了末尾则回到开头
        _project_index = (_project_index + 1) % len(GEMINI_PROJECT_IDS)
        
        return project_id
