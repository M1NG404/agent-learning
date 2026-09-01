tools = [
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "根据订单ID查询订单信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_order",
            "description": "根据订单ID取消订单",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "订单ID"
                    }
                },
                "required": ["order_id"]
            }
        }
    }
]
