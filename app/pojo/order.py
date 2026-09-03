from pydantic import BaseModel


class OrderInfo(BaseModel):
    orderId: int
    status: str
    amount: float

class OrderArgs(BaseModel):
    order_id: int