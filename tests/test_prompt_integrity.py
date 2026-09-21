"""What the agents are actually told.

Three problems the audit found: one analyst's brief reached the model as a Python
tuple, every analyst was asked for a trade call that nothing reads, and a report
that was never produced was presented as an empty labelled section, which invites
the next agent to fill it in from nothing.
"""

from __future__ import annotations

import importlib

import pytest

ANALYSTS = ["market_analyst", "sentiment_analyst", "news_analyst", "fundamentals_analyst"]


@pytest.mark.unit
@pytest.mark.parametrize("name", ANALYSTS)
def test_an_analyst_brief_is_text_not_a_python_object(name):
    """A trailing comma made one brief a tuple, so the model was handed its repr
    (quotes, parens and all) instead of the instruction."""
    import ast
    import inspect

    mod = importlib.import_module(f"tradingagents.agents.analysts.{name}")
    tree = ast.parse(inspect.getsource(mod))
    briefs = [node.value for node in ast.walk(tree)
              if isinstance(node, ast.Assign)
              and getattr(node.targets[0], "id", "") == "system_message"]
    assert briefs, f"{name} has no system_message"
    for brief in briefs:
        assert not isinstance(brief, ast.Tuple), "the brief is a tuple, not text"


@pytest.mark.unit
@pytest.mark.parametrize("name", ANALYSTS)
def test_an_analyst_is_not_asked_for_a_trade_call_nothing_reads(name):
    """The stop signal is never consumed, and asking for it makes an analyst
    open with a direction that then travels as evidence."""
    import inspect

    mod = importlib.import_module(f"tradingagents.agents.analysts.{name}")
    assert "FINAL TRANSACTION PROPOSAL" not in inspect.getsource(mod)


@pytest.mark.unit
@pytest.mark.parametrize("module, factory", [
    ("tradingagents.agents.researchers.bull_researcher", "create_bull_researcher"),
    ("tradingagents.agents.researchers.bear_researcher", "create_bear_researcher"),
    ("tradingagents.agents.risk_mgmt.aggressive_debator", "create_aggressive_debator"),
    ("tradingagents.agents.risk_mgmt.conservative_debator", "create_conservative_debator"),
    ("tradingagents.agents.risk_mgmt.neutral_debator", "create_neutral_debator"),
])
def test_a_report_that_was_never_produced_says_so(module, factory):
    """`--analysts market` leaves three reports empty; presenting them as blank
    sections invites the model to invent the contents."""
    from langchain_core.messages import AIMessage

    mod = importlib.import_module(module)
    seen = []

    class _LLM:
        def invoke(self, prompt, *a, **k):
            seen.append(prompt if isinstance(prompt, str) else str(prompt))
            return AIMessage("argument")

        def with_structured_output(self, *a, **k):
            raise NotImplementedError

    state = {
        "company_of_interest": "NVDA", "trade_date": "2026-08-14", "asset_type": "stock",
        "instrument_context": "", "portfolio_context": "", "past_context": "",
        "market_report": "RSI 61, price 178.", "sentiment_report": "", "news_report": "",
        "fundamentals_report": "", "investment_plan": "P", "trader_investment_plan": "T",
        "investment_debate_state": {"bull_history": "", "bear_history": "", "history": "",
                                    "current_response": "", "judge_decision": "", "count": 0},
        "risk_debate_state": {"history": "", "latest_speaker": "", "count": 0,
                              "aggressive_history": "", "conservative_history": "", "neutral_history": "",
                              "current_aggressive_response": "", "current_conservative_response": "",
                              "current_neutral_response": "", "judge_decision": ""},
    }
    getattr(mod, factory)(_LLM())(state)

    prompt = " ".join(seen)
    assert "not part of this run" in prompt or "not available" in prompt, prompt[:400]
    assert "RSI 61" in prompt  # the report that does exist is still passed through


@pytest.mark.unit
def test_the_fundamentals_analyst_is_told_to_read_both_frequencies():
    """Found in a live 6134 run against an annual-only filing-dated vendor.

    Both statement tools default to ``freq="quarterly"``, so whether the
    filing-dated series was read at all came down to which frequency the model
    happened to pick. On 7203 it asked for annual and got EDINET DB; on 6134 it
    asked for quarterly, EDINET DB declined (it serves annual only), yfinance
    served period-dated figures instead, and the analyst reported revenue and
    operating income as unavailable — for a year in which revenue rose 42% and
    operating income 112%.
    """
    import inspect

    from tradingagents.agents.analysts import fundamentals_analyst
    from tradingagents.agents.utils import fundamental_data_tools

    source = inspect.getsource(fundamentals_analyst.create_fundamentals_analyst)
    assert "freq='annual'" in source and "freq='quarterly'" in source
    assert "BOTH frequencies" in source

    # The tool's own parameter description has to carry the reason too: a model
    # reading only the signature must not take "quarterly" as the whole story.
    for tool in (fundamental_data_tools.get_balance_sheet,
                 fundamental_data_tools.get_cashflow,
                 fundamental_data_tools.get_income_statement):
        described = str(tool.args_schema.model_json_schema()["properties"]["freq"])
        assert "Read both" in described, tool.name
