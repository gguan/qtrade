# components/commissions.py

from abc import ABC, abstractmethod

class Commission(ABC):
    @abstractmethod
    def calculate_commission(self, order_size: int, fill_price: float) -> float:
        """
        Calculate the commission for an order.

        :param order_size: Order size (positive for buy, negative for sell)
        :param fill_price: Order fill price
        :return: Commission fee
        """
        pass

class NoCommission(Commission):
    def calculate_commission(self, order_size: int, fill_price: float) -> float:
        return 0.0

class PercentageCommission(Commission):
    def __init__(self, percentage: float):
        """
        Initialize the percentage commission scheme.

        :param percentage: Commission percentage (e.g., 0.001 for 0.1%)
        """
        self.percentage = percentage

    def calculate_commission(self, order_size: int, fill_price: float) -> float:
        return abs(order_size * fill_price * self.percentage)

class FixedCommission(Commission):
    def __init__(self, fixed_fee: float):
        """
        Initialize the fixed commission scheme.

        :param fixed_fee: Fixed commission fee per order
        """
        self.fixed_fee = fixed_fee

    def calculate_commission(self, order_size: int, fill_price: float) -> float:
        return self.fixed_fee

class SlippageCommission(Commission):
    def __init__(self, slippage_percentage: float):
        """
        Initialize the slippage commission scheme.

        :param slippage_percentage: Slippage percentage (e.g., 0.001 for 0.1%)
        """
        self.slippage_percentage = slippage_percentage

    def calculate_commission(self, order_size: int, fill_price: float) -> float:
        return abs(order_size * fill_price * self.slippage_percentage)