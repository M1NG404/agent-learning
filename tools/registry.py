from tools.order_tools import cancel_order, get_order


tool_registry = {
    "get_order": get_order,
    "cancel_order": cancel_order
}