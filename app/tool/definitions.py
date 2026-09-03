
# 给 LLM 看的 Tool Schema

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
    },{
        "type":"function",
        "function":{
            "name":"save_memory",
            "description":"保存用户的长期记忆",
            "parameters":{
                "type":"object",
                "properties":{
                    "key":{
                        "type":"string",
                        "description":"记忆字段名，如user_name, user_age等"
                    },
                    "value":{
                        "type":"string",
                        "description":"需要保存的记忆内容"
                    }
                },
                "required":["key", "value"]
            }
        }
    }
]
