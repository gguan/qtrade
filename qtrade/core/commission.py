# components/commission.py

from abc import ABC, abstractmethod


class Commission(ABC):
    """
    Abstract base class for different commission schemes.
    """

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
    """
    Commission scheme with no fees.
    """

    def calculate_commission(self, order_size: int, fill_price: float) -> float:
        """
        Returns zero commission.

        :param order_size: Order size (positive for buy, negative for sell)
        :param fill_price: Order fill price
        :return: 0.0
        """
        return 0.0

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PercentageCommission(Commission):
    """
    Commission scheme based on a percentage of the order value.
    """

    def __init__(self, percentage: float):
        """
        Initialize the percentage commission scheme.

        :param percentage: Commission percentage (e.g., 0.001 for 0.1%)
        :raises ValueError: If percentage is negative.
        """
        if percentage < 0:
            raise ValueError("Percentage cannot be negative.")
        self.percentage: float = percentage

    def calculate_commission(self, order_size: int, fill_price: float) -> float:
        """
        Calculate commission as a percentage of the order value.

        :param order_size: Order size (positive for buy, negative for sell)
        :param fill_price: Order fill price
        :return: Calculated commission fee
        """
        return abs(order_size * fill_price * self.percentage)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(percentage={self.percentage})"


class FixedCommission(Commission):
    """
    Commission scheme with a fixed fee per order.
    """

    def __init__(self, fixed_fee: float):
        """
        Initialize the fixed commission scheme.

        :param fixed_fee: Fixed commission fee per order
        :raises ValueError: If fixed_fee is negative.
        """
        if fixed_fee < 0:
            raise ValueError("Fixed fee cannot be negative.")
        self.fixed_fee: float = fixed_fee


    def calculate_commission(self, order_size: int, fill_price: float) -> float:
        """
        Calculate commission as a fixed fee per order.

        :param order_size: Order size (positive for buy, negative for sell)
        :param fill_price: Order fill price
        :return: Fixed commission fee
        """
        return self.fixed_fee

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(fixed_fee={self.fixed_fee})"


class SlippageCommission(Commission):
    """
    Commission scheme that accounts for slippage as a percentage of the order value.
    """

    def __init__(self, slippage_percentage: float):
        """
        Initialize the slippage commission scheme.

        :param slippage_percentage: Slippage percentage (e.g., 0.001 for 0.1%)
        :raises ValueError: If slippage_percentage is negative.
        """
        if slippage_percentage < 0:
            raise ValueError("Slippage percentage cannot be negative.")
        self.slippage_percentage: float = slippage_percentage

    def calculate_commission(self, order_size: int, fill_price: float) -> float:
        """
        Calculate slippage as a percentage of the order value.

        :param order_size: Order size (positive for buy, negative for sell)
        :param fill_price: Order fill price
        :return: Calculated slippage fee
        """
        return abs(order_size * fill_price * self.slippage_percentage)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(slippage_percentage={self.slippage_percentage})"
