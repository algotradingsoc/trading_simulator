import typing

price_data = typing.Tuple[float, float, float]  # (bid, ask, dividend)
price_dict = typing.Dict[str, price_data]
portfolio = typing.Dict[str, float]


def agent(ts: typing.List[price_dict],holdings: portfolio,) -> portfolio 
    orders = {'A':1}
    return orders 

def backtest(
    ts: typing.List[price_dict],
    agent,
    cash: float,
    holdings: portfolio,
    history: typing.List[float] = [],
) -> typing.List[float]:
    if not len(ts):
        return history
    day = ts[0]
    # TODO: agent can use data from previous days also (lookback period) 
    orders = agent(day, holdings)  # agent function returns dictionary of orders
    
    order_cost = sum([orders[x] * day[x][1] for x in orders])
    # check orders possible
    assert order_cost <= cash, "insufficent cash to enact order"
    dividends = [day[x][2] * holdings[x] for x in holdings]
    eod_holdings = dict(
        [(x, holdings[x] + (orders[x] if x in orders else 0)) for x in holdings]
    )
    eod_cash = cash + sum(dividends) - order_cost
    portfolio_value = eod_cash + sum(
        [eod_holdings[x] * day[x][0] for x in eod_holdings]
    )
    return backtest(ts[1:], agent, eod_cash, eod_holdings, history + [portfolio_value])


# an extremely basic example, one symbol, price never changes, and agent chooses to do nothing
print(
    backtest(
        [{"A": (1, 2, 0)} for x in range(900)], lambda x, y: {}, 500, {"A": 500}, []
    )
)
