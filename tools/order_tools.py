from models.order_model import OrderInfo


def get_order(order_id: int):
    order = OrderInfo(
        orderId=order_id,
        status="SHIPPED",
        amount=99.9
    )

    return order.model_dump()