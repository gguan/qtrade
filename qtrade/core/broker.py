# components/engine.py

import logging
from typing import List, Optional
from .trade import Trade
from .order import Order
from .position import Position
from .commission import Commission
import pandas as pd
import numpy as np

class Broker:
    """
    Broker类负责订单执行与仓位管理的逻辑。

    Attributes:
        data (pd.DataFrame): 包含行情数据的DataFrame，必须包含列['open', 'high', 'low', 'close']。
        cash (float): 当前账户的现金余额。
        commission (Optional[Commission]): 用于计算交易费用的类实例，如果为None则不收取佣金。
        margin_ratio (float): 保证金比例（0,1]区间，暂未使用。
        trade_on_close (bool): True则订单在当bar的close价执行，否则在下一bar的open价执行。
        position (Position): 当前持仓信息。
        current_time (pd.Timestamp): 当前所处的bar时间戳。
        completed_trades (List[Trade]): 当前bar已完成的交易列表。
        _new_orders (List[Order]): 本bar新提交的订单。
        pending_orders (List[Order]): 未完成的挂单，包括stop/limit订单。
        executing_orders (List[Order]): 若 trade_on_close=False，则需在下bar执行的订单列表。
        _order_history (List[Order]): 历史订单列表。
        _account_value_history (pd.Series): 记录历史账户价值的时间序列。

    Note:
        请确保 data.index 为时间类型索引，且包含 'open','high','low','close' 列。
    """
    def __init__(self, 
                 data: pd.DataFrame, 
                 cash: float, 
                 commission: Optional[Commission], 
                 margin_ratio: float, 
                 trade_on_close: bool):
        assert cash > 0, "Initial cash must be positive."
        assert 1 >= margin_ratio > 0, "Margin must be between 0 and 1"

        self.data = data
        self.cash = cash
        self.commission = commission
        self.margin_ratio = margin_ratio
        self.trade_on_close = trade_on_close
        self.position = Position()

        self.current_time = data.index[0]
        self.completed_trades: List[Trade] = [] # 储存本次bar已经完成的交易

        self._new_orders: List[Order] = [] # 本次bar新建的订单
        self._pending_orders: List[Order] = [] # 尚未执行的订单
        self._executing_orders: List[Order] = [] # 如果trade_on_close是false，正在执行的订单需要在下一个bar的open price成交

        self._order_history: List[Order] = []
        self._account_value_history = pd.Series(data=self.cash, index=data.index).astype('float64') 
    

    @property
    def account_value(self) -> float:
        return self.cash +  self.unrealized_pnl
    
    @property
    def cummulative_returns(self) -> float:
        return self.account_value / self._account_value_history.iloc[0]

    @property
    def required_margin(self) -> float:
        """
        计算当前账户的总保证金需求。
        """
        return sum(abs(trade.size) * trade.entry_price * self.margin_ratio for trade in self.position.active_trades)
    
    @property
    def unrealized_pnl(self) -> float:
        current_price = self.data.loc[self.current_time, 'close']
        return sum(trade.size * (current_price - trade.entry_price) for trade in self.position.active_trades) 


    @property
    def trade_history(self) -> List[Trade]:
        return self.position.closed_trades
    
    @property
    def order_history(self) -> List[Order]:
        return self._order_history
    
    @property
    def account_value_history(self):
        return self._account_value_history
    
    def process_bar(self, current_time: pd.Timestamp):
        self.current_time = current_time
        self.completed_trades.clear()
        self._check_sl_tp()
        self._process_pending_orders()


    def update_account_value_history(self):
        self._account_value_history.loc[self.current_time] = self.account_value

    def new_order(self, order: Order):
        self._new_orders.append(order)

    def new_orders(self, orders: List[Order]):
        self._new_orders.extend(orders)

    def _process_pending_orders(self):
        high, low = self.data.loc[self.current_time, 'high'], self.data.loc[self.current_time, 'low']
        
        # 1. 处理 "excuting_orders"
        # 这些订单是上一个step中已经确定要在下一个bar开盘价执行的订单（如市价单在trade_on_close=False时延迟到下一个bar的open执行）。
        for order in self._executing_orders:
            fill_date = self.current_time
            fill_price = self.data.loc[fill_date, 'open']
            self._process_order(order, fill_price, fill_date)
        self._executing_orders.clear()

        orders_to_remove = []
         # 2. 处理 "pending_orders"
        # pending_orders中存放的是之前bar留下的未触发或未执行的订单，包括stop、limit或等待下一个bar执行的订单。
        for order in self._pending_orders:
            # 如果订单有stop条件，先判断stop是否触发
            if order._stop:
                is_stop_triggered = high >= order._stop if order.is_long else low <= order._stop
                if is_stop_triggered:
                    # Stop 被触发，更新订单状态
                    order._stop = None
                else:
                    continue
            
            # 如果订单有limit价格，则检查limit是否触发
            if order._limit: 
                is_limit_triggered = low < order._limit if order.is_long else high > order._limit
                if is_limit_triggered:
                    fill_date = self.current_time
                    fill_price = order._limit
                    self._process_order(order, fill_price, fill_date)
                    orders_to_remove.append(order)
                else:
                    continue
            else: 
                # 没有limit的情况即市价单
                if self.trade_on_close:
                    fill_date = self.current_time
                    fill_price = self.data.loc[fill_date, 'close']
                    self._process_order(order, fill_price, fill_date)
                    orders_to_remove.append(order)
                else:
                    self._executing_orders.append(order)
        # 将orders_to_remove中的订单从pending_orders中移除                    
        for order in orders_to_remove:
            self._pending_orders.remove(order)


    def process_new_orders(self):
        for order in self._new_orders:
            if order._stop or order._limit:
                self._pending_orders.append(order)
            else:
                if self.trade_on_close:
                    fill_date = self.current_time
                    fill_price = self.data.loc[fill_date, 'close']
                    self._process_order(order, fill_price, fill_date)
                else:
                    self._executing_orders.append(order)
        self._new_orders.clear()


    def _process_order(self, order: Order, fill_price: float, fill_date: pd.Timestamp):
        """
        Process a filled order.

        :param order: Order to process
        :param fill_price: Price at which the order was filled
        :param fill_date: Date when the order was filled
        """
        if not self._is_margin_sufficient(order, fill_price):
            # 保证金不足，拒绝订单
            order.reject(reason="Insufficient margin")
            logging.info(f"Order rejected: {order._reject_reason}")
            self._order_history.append(order)
            return

        remaining_order_size = order._size  # Size needed to fulfill the order

        commission_cost = self.commission.calculate_commission(order._size, fill_price) if self.commission else 0
        self.cash -= commission_cost

        for trade in self.position.active_trades:
            if trade.is_long == order.is_long:
                continue
            if abs(remaining_order_size) >= abs(trade.size):
                # Close the existing trade
                closed_trade = self.position.close_position(trade, trade.size, fill_price, fill_date,'signal')
                self.completed_trades.append(closed_trade)
                self.cash += closed_trade.profit
                remaining_order_size += closed_trade.size
            else:
                # Partially close the trade
                closed_trade = self.position.close_position(trade, -remaining_order_size, fill_price, fill_date, 'signal')
                self.completed_trades.append(closed_trade)
                self.cash += closed_trade.profit
                remaining_order_size = 0
            if remaining_order_size == 0:
                break
        
        # remove from position.active_trades where size == 0
        self.position.active_trades = [trade for trade in self.position.active_trades if trade.size != 0]
        
        if remaining_order_size:
            self.position.open_position(fill_price, fill_date, remaining_order_size, order._sl, order._tp, order.tag)
       
        # 记录订单
        order.fill(fill_price, fill_date)
        self._order_history.append(order)

    def _is_margin_sufficient(self, order: Order, fill_price: float) -> bool:
        """
        Check if there is sufficient margin to execute the order.

        :param order: Order to check
        :return: True if there is sufficient margin
        """
        new_position_size = self.position.size + order.size
        new_margin = abs(new_position_size) * fill_price * self.margin_ratio

        unrealized_pnl = sum(trade.size * (fill_price - trade.entry_price) for trade in self.position.active_trades) 
        account_value = self.cash + unrealized_pnl
        
        return account_value >= new_margin
        
    def _check_sl_tp(self):
        """
        Check and apply stop loss and take profit.
        """
        high, low = self.data.loc[self.current_time, 'high'], self.data.loc[self.current_time, 'low']
        for trade in self.position.active_trades:
            if not trade.sl and not trade.tp:
                continue
            
            sl = trade.sl
            tp = trade.tp
            if trade.is_long:
                if sl is not None and low <= sl:
                    # Stop loss hit
                    commission_cost = self.commission.calculate_commission(abs(trade.size), sl) if self.commission else 0
                    self.cash -= commission_cost
                    closed_trade = self.position.close_position(
                        trade=trade,
                        close_size=trade.size,
                        exit_price=sl,
                        exit_date=self.current_time,
                        exit_reason='sl'
                    )
                    self.completed_trades.append(closed_trade)
                    self.cash += closed_trade.profit
                elif tp is not None and high >= tp:
                    # Take profit hit
                    commission_cost = self.commission.calculate_commission(abs(trade.size), tp) if self.commission else 0
                    self.cash -= commission_cost
                    closed_trade = self.position.close_position(
                        trade=trade,
                        close_size=trade.size,
                        exit_price=tp,
                        exit_date=self.current_time,
                        exit_reason='tp'
                    )
                    self.completed_trades.append(closed_trade)
                    self.cash += closed_trade.profit
            else:
                if sl is not None and high >= sl:
                    # Stop loss hit
                    commission_cost = self.commission.calculate_commission(abs(trade.size), sl) if self.commission else 0
                    self.cash -= commission_cost
                    closed_trade = self.position.close_position(
                        trade=trade,
                        close_size=trade.size,
                        exit_price=sl,
                        exit_date=self.current_time,
                        exit_reason='sl'
                    )
                    self.completed_trades.append(closed_trade)
                    self.cash += closed_trade.profit
                elif tp is not None and low <= tp:
                    # Take profit hit
                    commission_cost = self.commission.calculate_commission(abs(trade.size), tp) if self.commission else 0
                    self.cash -= commission_cost
                    closed_trade = self.position.close_position(
                        trade=trade,
                        close_size=trade.size,
                        exit_price=tp,
                        exit_date=self.current_time,
                        exit_reason='tp'
                    )
                    self.completed_trades.append(closed_trade)
                    self.cash += closed_trade.profit
        # remove from position.active_trades where size == 0
        self.position.active_trades = [trade for trade in self.position.active_trades if trade.size != 0]


    def close_all_positions(self):
        """
        Close all open positions at the end of the episode.
        """
        price = self.data.loc[self.current_time, 'close']
        for trade in self.position.active_trades.copy():
            commission_cost = self.commission.calculate_commission(abs(trade.size), price) if self.commission else 0
            self.cash -= commission_cost
            closed_trade = self.position.close_position(
                trade=trade,
                close_size=trade.size,
                exit_price=price,
                exit_date=self.current_time,
                exit_reason='end'
            )
            self.completed_trades.append(closed_trade)
            self.cash += closed_trade.profit
        # remove from position.active_trades where size == 0
        self.position.active_trades = [trade for trade in self.position.active_trades if trade.size != 0]