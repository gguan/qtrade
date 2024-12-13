# tests/test_broker.py

import pytest
import pandas as pd
from qtrade.core import Broker,Order
from qtrade.core import NoCommission, PercentageCommission


@pytest.fixture
def sample_data():
    """
    创建一个包含5个交易日的模拟市场数据:
    日间价格平稳增长，方便计算与检查。
    """
    data = pd.DataFrame({
        'open': [100, 102, 104, 106, 108],
        'high': [105, 107, 109, 111, 113],
        'low': [96, 97, 99, 101, 103],
        'close': [102, 104, 106, 108, 110]
    }, index=pd.to_datetime(['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04', '2024-01-05']))
    return data

@pytest.fixture
def broker_no_commission(sample_data):
    commission = NoCommission()
    broker = Broker(
        data=sample_data,
        cash=10000.0,
        commission=commission,
        margin_ratio=0.1,
        trade_on_close=True
    )
    return broker

@pytest.fixture
def broker_percentage_commission(sample_data):
    commission = PercentageCommission(percentage=0.001)  # 0.1%
    broker = Broker(
        data=sample_data,
        cash=10000.0,
        commission=commission,
        margin_ratio=0.1,
        trade_on_close=True
    )
    return broker

@pytest.fixture
def broker_no_commission_trade_on_open(sample_data):
    commission = NoCommission()
    broker = Broker(
        data=sample_data,
        cash=10000.0,
        commission=commission,
        margin_ratio=0.1,
        trade_on_close=False  # 设置 trade_on_close=False
    )
    return broker

def test_broker_initial_state(broker_no_commission):
    """测试 Broker 初始状态，包括 _equity_history、可用保证金、权益等。"""
    assert broker_no_commission.cash == 10000.0
    assert broker_no_commission.position.size == 0
    assert broker_no_commission.equity == 10000.0
    assert broker_no_commission.available_margin == 10000.0
    assert broker_no_commission.unrealized_pnl == 0.0
    assert len(broker_no_commission.closed_trades) == 0
    assert broker_no_commission.filled_orders == ()
    assert broker_no_commission.closed_orders == ()
    # 初始 equity_history 应该是与 cash 相同
    assert broker_no_commission.equity_history.iloc[0] == 10000.0

def test_equity_history_update(broker_no_commission):
    """在不进行任何交易情况下，多日调用 process_bar，测试 equity_history 更新。"""
    # 初始为第一天
    assert broker_no_commission.equity_history.iloc[0] == 10000.0
    # 处理第二天
    broker_no_commission.process_bar(pd.Timestamp('2024-01-02'))
    # 第二天末尾应更新 equity_history
    assert broker_no_commission.equity_history.loc[pd.Timestamp('2024-01-02')] == broker_no_commission.equity
    assert broker_no_commission.equity == 10000.0

def test_open_position_and_equity_history(broker_no_commission):
    """测试 open_position 后 equity_history 的变化。
       第一天处理后检查未平仓盈亏，第二天价格上涨后再检查。"""
    # 打开一个仓位：多头 10 手，价格102进场
    broker_no_commission._open_trade(
        entry_price=102.0,
        entry_date=pd.Timestamp('2024-01-01'),
        size=10,
        sl=None,
        tp=None,
        tag="Pos1"
    )
    # 处理第一天
    broker_no_commission.process_bar(pd.Timestamp('2024-01-01'))
    # 未平仓盈亏=(102-102)*10=0，权益不变
    assert broker_no_commission.equity == 10000.0
    assert broker_no_commission.equity_history.loc['2024-01-01'] == 10000.0

    # 第二天价格=104, 未平仓盈亏=(104-102)*10=20
    broker_no_commission.process_bar(pd.Timestamp('2024-01-02'))
    assert broker_no_commission.unrealized_pnl == 20.0
    assert broker_no_commission.equity == 10020.0
    assert broker_no_commission.equity_history.loc['2024-01-02'] == 10020.0

def test_close_position_and_equity(broker_no_commission):
    """打开仓位后第二天获利平仓，检查 equity、equity_history、closed_trades 的正确性。"""
    # 打开多头10手，102进场
    broker_no_commission._open_trade(
        entry_price=102.0,
        entry_date=pd.Timestamp('2024-01-01'),
        size=10,
        sl=None,
        tp=None,
        tag="Pos1"
    )
    broker_no_commission.process_bar(pd.Timestamp('2024-01-01'))

    # 第二天价格104, 未平仓盈亏20
    broker_no_commission.process_bar(pd.Timestamp('2024-01-02'))
    assert broker_no_commission.equity == 10020.0

    # 使用 close_all_positions 平仓
    broker_no_commission.close_all_positions()
    # 平仓价108（假设当天为2024-01-02结束价）
    # 实际情况请根据broker的逻辑修改，如果close_all_positions在2024-01-02处理后才执行，则exit_price为104还是108需明确
    # 假设close_all_positions是在处理bar后执行，并使用当前bar的close价 104
    # 利润=(104-102)*10=20
    assert broker_no_commission.equity == 10020.0
    assert len(broker_no_commission.position.closed_trades) == 1
    closed_trade = broker_no_commission.position.closed_trades[0]
    assert closed_trade.profit == 20.0
    # check equity_history
    assert broker_no_commission.equity_history.loc['2024-01-02'] == 10020.0

def test_place_and_reject_order_due_to_margin(broker_no_commission):
    """测试当保证金不足时，订单被正确拒绝，_equity_history 和 账户状态不应改变。"""
    # 极大size超出保证金范围
    order = Order(size=2000, limit=None, stop=None, sl=95.0, tp=105.0, tag="HugeOrder")
    broker_no_commission.place_orders(order)
    broker_no_commission.process_bar(pd.Timestamp('2024-01-01'))

    # 检查订单是否被拒绝
    assert broker_no_commission.filled_orders == ()
    assert len(broker_no_commission.closed_orders) == 1
    rejected_order = broker_no_commission.closed_orders[0]
    assert rejected_order == order
    assert rejected_order._close_reason == "Insufficient margin"
    # 账户状态不应变
    assert broker_no_commission.cash == 10000.0
    assert broker_no_commission.position.active_trades == ()
    assert broker_no_commission.equity_history.loc['2024-01-01'] == 10000.0

def test_multiple_orders_and_positions(broker_percentage_commission):
    """
    测试在有百分比佣金的情况下提交多个订单。
    分别提交一个多头市场单和一个空头限价单。
    检查佣金扣除、_equity_history、可用保证金和未平仓盈亏变化。
    """

    # 第一天：执行market order1
    broker_percentage_commission.process_bar(pd.Timestamp('2024-01-01'))
    order1 = Order(size=10, limit=None, stop=None, sl=95.0, tp=108.0, tag="MktOrderLong")
    order2 = Order(size=-5, limit=103.0, stop=None, sl=90.0, tp=110.0, tag="LimitOrderShort")
    broker_percentage_commission.place_orders([order1, order2])

    # 预期佣金=10*102*0.001=1.02
    expected_commission_1 = 1.02
    assert broker_percentage_commission.cash == 10000.0 - expected_commission_1
    assert len(broker_percentage_commission.position.active_trades) == 1
    active_trade1 = broker_percentage_commission.position.active_trades[0]
    assert active_trade1.size == 10
    assert broker_percentage_commission.position.size == 10
    assert active_trade1.entry_price == 102.0
    assert broker_percentage_commission.equity_history.loc['2024-01-01'] == broker_percentage_commission.equity

    # 第二天：触发limit order2
    broker_percentage_commission.process_bar(pd.Timestamp('2024-01-02'))
    # order2填单价=103，佣金=5*103*0.001=0.45
    expected_commission_2 = 0.515
    total_commission = expected_commission_1 + expected_commission_2
    profit1 = broker_percentage_commission.position.closed_trades[0].profit
    assert broker_percentage_commission.cash == 10000.0 - total_commission + profit1
    assert len(broker_percentage_commission.position.active_trades) == 1
    active_trade2 = broker_percentage_commission.position.active_trades[0]
    assert active_trade2.size == 5

    # 检查第二天末尾的 equity
    # 未平仓盈亏=长头(5手*(104-102)=10) + 平仓盈亏=短头(5手*(103-102)=5)
    # 总权益=10000-1.02-0.45+10+5=10013.53
    assert pytest.approx(broker_percentage_commission.equity, 0.01) == 10013.53
    assert broker_percentage_commission.equity_history.loc['2024-01-02'] == pytest.approx(10013.53, 0.01)
    
    # 检查第三天末尾的 equity
    broker_percentage_commission.process_bar(pd.Timestamp('2024-01-03'))
    # tp commission 108 * 5 * 0.001 = 0.54
    expected_commission_3 = 0.54
    total_commission = expected_commission_1 + expected_commission_2 + expected_commission_3
    profit2 = broker_percentage_commission.position.closed_trades[1].profit
    assert broker_percentage_commission.equity_history.loc['2024-01-03'] == broker_percentage_commission.cash == 10000.0 - total_commission + profit1 + profit2

def test_sl_tp_trigger_for_long(broker_no_commission):
    """
    测试多头仓位在价格达到TP时是否正确触发平仓并更新账户属性。
    """
    # 开多头仓位
    broker_no_commission._open_trade(
        entry_price=100.0,
        entry_date=pd.Timestamp('2024-01-01'),
        size=10,
        sl=94.0,
        tp=105.0,
        tag="SLTPTrade"
    )
    broker_no_commission.process_bar(pd.Timestamp('2024-01-01'))
    # 第二天价格104，无触发SL/TP
    broker_no_commission.process_bar(pd.Timestamp('2024-01-02'))
    # 第三天价格106, 已超过TP=105, 触发平仓
    broker_no_commission.process_bar(pd.Timestamp('2024-01-03'))

    # 检查仓位已关闭
    assert broker_no_commission.position.active_trades == ()
    assert len(broker_no_commission.position.closed_trades) == 1
    closed_trade = broker_no_commission.position.closed_trades[0]
    # 平仓价格应为当日close=105
    assert closed_trade.exit_reason == 'tp'
    assert closed_trade.profit == 50
    # 检查账户现金
    assert broker_no_commission.cash == 10000.0 + 50.0
    # 检查equity_history
    # 第三天末尾权益=10060
    assert broker_no_commission.equity_history.loc['2024-01-03'] == 10050.0

def test_sl_tp_trigger_for_short(broker_no_commission):
    """
    测试空头仓位在价格达到SL时是否正确触发平仓并更新账户属性。
    """
    # 开空头仓位
    broker_no_commission._open_trade(
        entry_price=104.0,
        entry_date=pd.Timestamp('2024-01-01'),
        size=-10,
        sl=110.0,
        tp=95.0,
        tag="ShortSLTP"
    )
    broker_no_commission.process_bar(pd.Timestamp('2024-01-01'))
    assert len(broker_no_commission.position.closed_trades) == 0
    # 第二天价格104，无触发SL/TP
    broker_no_commission.process_bar(pd.Timestamp('2024-01-02'))
    broker_no_commission.unrealized_pnl == 0
    # 第三天价格106, 无触发
    broker_no_commission.process_bar(pd.Timestamp('2024-01-03'))
    broker_no_commission.unrealized_pnl == -20
    # 第四天价格110，触发SL
    broker_no_commission.process_bar(pd.Timestamp('2024-01-04'))

    # 检查仓位已关闭
    assert broker_no_commission.position.active_trades == ()
    assert len(broker_no_commission.position.closed_trades) == 1
    closed_trade = broker_no_commission.position.closed_trades[0]
    assert closed_trade.exit_reason == 'sl'
    # 利润=(entry_price - exit_price)*abs(size)=(104-110)*10=-60
    assert closed_trade.profit == -60
    assert broker_no_commission.cash == 10000.0 - 60.0
    # 第四天末尾权益=9940
    assert broker_no_commission.equity_history.loc['2024-01-04'] == 9940.0

def test_process_new_orders_direct_call(broker_no_commission):
    """
    测试直接调用process_new_orders逻辑，当有new_orders但未调用process_bar。
    如果trade_on_close=True，应当直接执行订单。
    """
    order = Order(size=10, limit=None, stop=None, sl=None, tp=None, tag="MktDirect")
    broker_no_commission.place_orders(order)

    # trade_on_close=True，当日close=102,直接执行订单
    assert broker_no_commission.filled_orders == (order,)
    assert len(broker_no_commission.position.active_trades) == 1
    active_trade = broker_no_commission.position.active_trades[0]
    assert active_trade.entry_price == 102.0
    # equity_history不会在process_new_orders时更新，因为更新在process_bar执行
    # 但可以检查未处理bar时状态不变
    assert broker_no_commission.equity == 10000.0

def test_open_close_multiple_positions_across_days(broker_no_commission):
    """
    测试跨多日开仓和平仓多个仓位，并检查每个bar末尾equity_history的更新情况。
    """
    # 第一天：无交易
    broker_no_commission.process_bar(pd.Timestamp('2024-01-01'))
    assert broker_no_commission.equity_history.loc['2024-01-01'] == 10000.0

    # 第二天：开一个多头
    broker_no_commission._open_trade(102.0, pd.Timestamp('2024-01-02'), 10)
    broker_no_commission.process_bar(pd.Timestamp('2024-01-02'))
    # 多头10手 @102, 第二日close=104
    # unrealized_pnl=(104-102)*10=20
    assert broker_no_commission.equity == 10020.0
    assert broker_no_commission.equity_history.loc['2024-01-02'] == 10020.0

    # 第三天：开一个空头 -5手 @106
    broker_no_commission._open_trade(106.0, pd.Timestamp('2024-01-03'), -5)
    broker_no_commission.process_bar(pd.Timestamp('2024-01-03'))
    # 多头未平(10手@102, close=106, pnl=40), 空头(-5手@106, close=106, pnl=0)
    # total_pnl=40, equity=10000+40=10040
    assert broker_no_commission.equity == 10040.0
    assert broker_no_commission.equity_history.loc['2024-01-03'] == 10040.0

    # 第四天：价格108
    broker_no_commission.process_bar(pd.Timestamp('2024-01-04'))
    # 多头pnl=(108-102)*10=60, 空头pnl=(106-108)*5=-10, total_pnl=50
    assert broker_no_commission.equity == 10050.0
    assert broker_no_commission.equity_history.loc['2024-01-04'] == 10050.0

    # 第五天：close_all_positions
    broker_no_commission.process_bar(pd.Timestamp('2024-01-05'))
    broker_no_commission.close_all_positions()
    # 第五天close=110
    # 多头profit=(110-102)*10=80, 空头profit=(106-110)*5=-20, net=60
    # equity=10000+60=10060
    assert broker_no_commission.equity == 10060.0
    assert broker_no_commission.equity_history.loc['2024-01-05'] == 10060.0
    assert broker_no_commission.position.active_trades == ()
    assert len(broker_no_commission.position.closed_trades) == 2


def test_margin_and_equity_updates(broker_percentage_commission):
    """
    测试在开仓、佣金扣除、价格变动后 margin 与 equity 的更新。
    数据已更新:
        'low' 在2024-01-01从95改为96
    """
    # 第1天: 2024-01-01, 无交易
    broker_percentage_commission.process_bar(pd.Timestamp('2024-01-01'))
    # equity = 10000.0
    assert broker_percentage_commission.equity == 10000.0, "初始equity应为10000.0"
    # equity_history 更新
    assert broker_percentage_commission.equity_history.loc['2024-01-01'] == 10000.0, "equity_history在2024-01-01应为10000.0"

    # 第2天: 2024-01-02, 开多头10手市价单
    broker_percentage_commission.process_bar(pd.Timestamp('2024-01-02'))
    order = Order(size=10, limit=None, stop=None, sl=None, tp=None, tag="Mkt2ndDay")
    broker_percentage_commission.place_orders(order)
    
    # 订单在2024-01-02的close价格104被填充
    # 佣金 = 10 * 104 * 0.001 = 1.04
    # cash = 10000.0 - 1.04 = 9998.96
    assert broker_percentage_commission.cash == pytest.approx(9998.96, 0.01), "佣金扣除后现金应为9998.96"
    # unrealized_pnl = (当前价格 - 进场价格) * size = (104 - 104) * 10 = 0
    assert broker_percentage_commission.unrealized_pnl == 0.0, "未实现盈亏应为0.0"
    # equity = cash + unrealized_pnl = 9998.96 + 0 = 9998.96
    assert broker_percentage_commission.equity == pytest.approx(9998.96, 0.01), "equity应为9998.96"
    # equity_history 更新
    assert broker_percentage_commission.equity_history.loc['2024-01-02'] == pytest.approx(9998.96, 0.01), "equity_history在2024-01-02应为9998.96"

    # available_margin = equity - used_margin
    # used_margin = abs(10) * 104 * 0.1 = 104
    # available_margin = 9998.96 - 104 = 9894.96
    expected_available_margin_day2 = 9998.96 - (10 * 104 * 0.1)  # 9894.96
    assert broker_percentage_commission.available_margin == pytest.approx(expected_available_margin_day2, 0.01), "available_margin应为9894.96"

    # 第3天: 2024-01-03, 价格=106
    broker_percentage_commission.process_bar(pd.Timestamp('2024-01-03'))
    # unrealized_pnl = (106 - 104) * 10 = 20
    assert broker_percentage_commission.unrealized_pnl == 20.0, "未实现盈亏应为20.0"
    # equity = cash + unrealized_pnl = 9998.96 + 20 = 10018.96
    assert broker_percentage_commission.equity == pytest.approx(10018.96, 0.01), "equity应为10018.96"
    # equity_history 更新
    assert broker_percentage_commission.equity_history.loc['2024-01-03'] == pytest.approx(10018.96, 0.01), "equity_history在2024-01-03应为10018.96"

    # available_margin = equity - used_margin = 10018.96 - 104 = 9914.96
    expected_available_margin_day3 = 10018.96 - (10 * 104 * 0.1)  # 9914.96
    assert broker_percentage_commission.available_margin == pytest.approx(expected_available_margin_day3, 0.01), "available_margin应为9914.96"

    # 第4天: 2024-01-04, 价格=108
    broker_percentage_commission.process_bar(pd.Timestamp('2024-01-04'))
    # unrealized_pnl = (108 - 104) * 10 = 40
    assert broker_percentage_commission.unrealized_pnl == 40.0, "未实现盈亏应为40.0"
    # equity = cash + unrealized_pnl = 9998.96 + 40 = 10038.96
    assert broker_percentage_commission.equity == pytest.approx(10038.96, 0.01), "equity应为10038.96"
    # available_margin = equity - used_margin = 10038.96 - 104 = 9934.96
    expected_available_margin_day4 = 10038.96 - (10 * 104 * 0.1)  # 9934.96
    assert broker_percentage_commission.available_margin == pytest.approx(expected_available_margin_day4, 0.01), "available_margin应为9934.96"

    # close_all_positions at 2024-01-04, 使用当天的close价格108
    broker_percentage_commission.close_all_positions()
    # profit = (108 - 104) * 10 = 40
    # cash = 9998.96 + 40 = 10038.96
    assert broker_percentage_commission.cash == pytest.approx(10038.96, 0.01), "平仓后现金应为10038.96"
    # equity = cash + unrealized_pnl = 10038.96 + 0 = 10038.96
    assert broker_percentage_commission.equity == pytest.approx(10038.96, 0.01), "平仓后equity应为10038.96"
    # equity_history 更新
    assert broker_percentage_commission.equity_history.loc['2024-01-04'] == pytest.approx(10038.96, 0.01), "equity_history在2024-01-04应为10038.96"
    # position 应该已清空
    assert broker_percentage_commission.position.active_trades == (), "所有仓位应已平仓"
    # closed_trades 数量为1
    assert len(broker_percentage_commission.position.closed_trades) == 1, "closed_trades应包含1个交易"
    # 检查关闭的交易详情
    closed_trade = broker_percentage_commission.position.closed_trades[0]
    assert closed_trade.size == 10, "关闭的交易size应为10"
    assert closed_trade.entry_price == 104.0, "关闭的交易entry_price应为104.0"
    assert closed_trade.exit_price == 108.0, "关闭的交易exit_price应为108.0"
    assert closed_trade.profit == 40.0, "关闭的交易profit应为40.0"

def test_order_fill_reject_sequence(broker_no_commission):
    """
    测试订单已填充后再次拒绝是否引发错误，已拒绝后再填充是否引发错误。
    """
    # 下一个市场单
    broker_no_commission.process_bar(pd.Timestamp('2024-01-01'))
    order = Order(size=10, limit=None, stop=None, sl=None, tp=None, tag="SeqOrder")
    broker_no_commission.place_orders(order)

    # 已填充
    assert order.is_filled is True

    # 尝试拒绝已填充的订单，应报错
    with pytest.raises(Exception):
        order._close(reason="Cannot reject a filled order.")

    # 新订单 过大，保证金不足被拒绝
    broker_no_commission.process_bar(pd.Timestamp('2024-01-02'))
    order2 = Order(size=2000, limit=None, stop=None, sl=None, tp=None, tag="Rejected")
    broker_no_commission.place_orders(order2)

    # 已被拒绝
    assert order2.is_closed is True

    # 尝试填充已拒绝的订单应报错
    with pytest.raises(Exception):
        order2._fill(110.0, pd.Timestamp('2024-01-03'))

# tests/test_broker.py

def test_trade_on_close_false_execution(broker_no_commission_trade_on_open):
    """
    测试 trade_on_close=False 时，订单在下一个 Bar 的开盘价执行。
    """
    broker = broker_no_commission_trade_on_open
    
    # 处理第1天的 Bar
    broker.process_bar(pd.Timestamp('2024-01-01'))
    # 第1天: 2024-01-01, 提交一个多头市场订单
    order = Order(size=10, limit=None, stop=None, sl=None, tp=None, tag="MktOrder1")
    broker.place_orders(order)
    
    
    # 此时订单应已提交，但尚未执行
    assert order not in broker.filled_orders, "订单在当前 Bar 未执行"
    assert order in broker._executing_orders or order in broker._pending_orders, "订单应在执行队列或待执行队列中"
    
    # 处理第2天的 Bar，订单应在第2天的开盘价执行
    broker.process_bar(pd.Timestamp('2024-01-02'))
    
    # 订单应被填充在第2天的开盘价102
    assert order in broker.filled_orders, "订单应已被填充"
    filled_order = broker.filled_orders[0]
    assert filled_order.fill_price == 102.0, f"订单应在开盘价102执行, 实际填充价为{filled_order.fill_price}"
    
    # 检查 cash 是否正确扣除佣金
    expected_commission = 10 * 102 * 0.001  # 1.02
    assert pytest.approx(broker.cash, 0.01) == 10000.0 - expected_commission, "现金应扣除佣金1.02"
    
    # 检查 equity_history
    assert broker.equity_history.loc['2024-01-02'] == pytest.approx(broker.equity, 0.01), "equity_history 应更新为当前 equity"
    
    # 检查 active trades
    assert len(broker.position.active_trades) == 1, "应有一个活跃仓位"
    active_trade = broker.position.active_trades[0]
    assert active_trade.size == 10, "仓位 size 应为10"
    assert active_trade.entry_price == 102.0, "仓位进场价应为102.0"
    
    # 检查 unrealized_pnl
    # 当日收盘价=104, unrealized_pnl=(104 - 102) * 10 = 20
    assert broker.unrealized_pnl == 20.0, "未实现盈亏应为20.0"
    
    # 检查 equity
    assert pytest.approx(broker.equity, 0.01) == 10000.0 - expected_commission + 20.0, "equity 应为10018.98"


def test_cancel_order(broker_no_commission):
    """
    测试取消订单的逻辑。
    """
    broker_no_commission.process_bar(pd.Timestamp('2024-01-01'))
    # 提交一个订单
    order = Order(size=10, limit=50, stop=None, sl=None, tp=None, tag="CancelOrder")
    broker_no_commission.place_orders(order)
    assert len(broker_no_commission._pending_orders) == 1, "应有一个执行中订单"

    broker_no_commission.process_bar(pd.Timestamp('2024-01-02'))
    assert len(broker_no_commission._pending_orders) == 1, "应有一个执行中订单"
    
    # 取消订单
    order.cancel()

    broker_no_commission.process_bar(pd.Timestamp('2024-01-03'))

    # 检查订单状态
    assert order.is_closed is True, "订单应为无效"
    assert order._close_reason == "Order canceled.", "订单拒绝原因应为'Order canceled.'"
    assert order in broker_no_commission._closed_orders, "订单应在关闭订单列表中"
    assert order not in broker_no_commission.filled_orders, "订单不应在已填充订单列表中"
    assert order not in broker_no_commission._executing_orders, "订单不应在执行订单列表中"
    assert order not in broker_no_commission._pending_orders, "订单不应在待执行订单列表中"
    
        # 检查订单是否可以再次取消
    with pytest.raises(Exception):
        order._fill(100.0, pd.Timestamp('2024-01-01'))
    # 检查订单是否已被填充
    assert order.is_filled is False, "已取消的订单不应被填充"
    assert order.fill_price is None, "已取消的订单不应有填充价格"
    
    # 检查订单是否可以再次取消
    with pytest.raises(Exception):
        order.cancel()