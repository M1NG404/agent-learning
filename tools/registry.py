
# 给 Runtime 用的 Tool 注册表

from models.order import OrderArgs
from tools.order_tools import cancel_order, get_order


tool_registry = {
    "get_order": {
        "function": get_order,
        "args_model": OrderArgs
    },
    "cancel_order": {
        "function": cancel_order,
        "args_model": OrderArgs
    }
}