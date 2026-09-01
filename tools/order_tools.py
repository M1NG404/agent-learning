from models.order import OrderInfo


def get_order(order_id: int):
    order = OrderInfo(
        orderId=order_id,
        status="SHIPPED",
        amount=99.9
    )

    return order.model_dump()

def cancel_order(order_id: int):
    return {
        "orderId": order_id,
        "status": "CANCELLED"
    }