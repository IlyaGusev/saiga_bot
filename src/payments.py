import json
import asyncio
import uuid
from enum import Enum

from yookassa import Configuration, Payment


class YookassaStatus(str, Enum):
    PENDING = "pending"
    WAITING_FOR_CAPTURE = "waiting_for_capture"
    SUCCEEDED = "succeeded"
    CANCELED = "canceled"


class YookassaHandler:
    def __init__(self, account_id: int, secret_key: str) -> None:
        Configuration.account_id = account_id
        Configuration.secret_key = secret_key

    def create_payment(self, value: int, description: str, email: str, bot_username: str):
        payment = Payment.create(
            {
                "amount": {"value": value, "currency": "RUB"},
                "confirmation": {"type": "redirect", "return_url": f"https://t.me/{bot_username}"},
                "capture": True,
                "description": description,
                "receipt": {
                    "customer": {"email": email},
                    "items": [
                        {
                            "description": description,
                            "amount": {"value": value, "currency": "RUB"},
                            "quantity": 1,
                            "vat_code": 1,
                        }
                    ],
                },
            }
        )
        return json.loads(payment.json())

    def cancel_payment(self, payment_id: str) -> None:
        idempotence_key = str(uuid.uuid4())
        Payment.cancel(payment_id, idempotence_key)

    def check_payment(self, payment_id: str) -> YookassaStatus:
        payment = json.loads((Payment.find_one(payment_id)).json())
        return YookassaStatus(payment["status"])
