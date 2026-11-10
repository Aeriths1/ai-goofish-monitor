import asyncio
import json
from types import SimpleNamespace

import pytest
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

import src.scraper as scraper
from src.failure_guard import FailureGuard
from src.services.search_pagination import PageAdvanceResult

FORBIDDEN_LAUNCH_FLAGS = (
    "--disable-web-security",
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-setuid-sandbox",
)


def test_user_validation_block_reason_detects_equivalent_payloads():
    cases = [
        (
            {
                "ret": [
                    "FAIL_SYS_USER_VALIDATE::哎哟喂,被挤爆啦,请稍后重试",
                    "RGV587_ERROR::SM::哎哟喂,被挤爆啦,请稍后重试",
                ],
                "data": {},
            },
            "FAIL_SYS_USER_VALIDATE",
        ),
        (
            {"ret": ["RGV587_ERROR::SM::哎哟喂,被挤爆啦,请稍后重试"], "data": {}},
            "RGV587_ERROR",
        ),
        (
            {"ret": "被挤爆啦,请稍后重试", "data": {}},
            "RGV587_ERROR",
        ),
        (
            {
                "ret": ["SUCCESS::调用成功"],
                "data": {"url": "https://example.test/punish?x5secdata=abc"},
            },
            "FAIL_SYS_USER_VALIDATE",
        ),
        (
            {
                "ret": ["SUCCESS::调用成功"],
                "data": {"url": "https://h5api.m.goofish.com/_____tmd_____/punish"},
            },
            "FAIL_SYS_USER_VALIDATE",
        ),
    ]
    for payload, reason in cases:
        assert scraper.user_validation_block_reason(payload) == reason


def test_user_validation_block_reason_ignores_ordinary_detail_payloads():
    assert scraper.user_validation_block_reason(None) is None
    assert scraper.user_validation_block_reason(["FAIL_SYS_USER_VALIDATE"]) is None
    assert (
        scraper.user_validation_block_reason(
            {"ret": ["FAIL_BIZ_ITEM_OFFLINE::商品下架"], "data": {"itemDO": {}}}
        )
        is None
    )
    assert (
        scraper.user_validation_block_reason(
            {
                "ret": ["SUCCESS::调用成功"],
                "data": {
                    "itemDO": {"title": "标题里写了 FAIL_SYS_USER_VALIDATE 也不算拦截"},
                    "sellerDO": {},
                },
            }
        )
        is None
    )


def test_abort_logs_terminate_and_raises(monkeypatch, capsys):
    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(scraper.asyncio, "sleep", _sleep)

    async def _run():
        with pytest.raises(scraper.RiskControlError, match="FAIL_SYS_USER_VALIDATE"):
            await scraper._abort_for_user_validation("FAIL_SYS_USER_VALIDATE")

    asyncio.run(_run())
    output = capsys.readouterr().out
    assert "程序将终止" in output
    assert "长时间休眠结束，现在将安全退出。" in output


class _Guard:
    threshold = 3

    def __init__(self):
        self.failures = []
        self.successes = []

    def should_skip_start(self, *_args, **_kwargs):
        return SimpleNamespace(
            skip=False,
            should_notify=False,
            reason="not_paused",
            paused_until=None,
            consecutive_failures=0,
        )

    def record_failure(self, task_name, reason, **kwargs):
        self.failures.append((task_name, reason, kwargs))
        return {
            "should_notify": False,
            "consecutive_failures": 1,
            "paused_until": None,
        }

    def record_success(self, task_name):
        self.successes.append(task_name)


class _Response:
    def __init__(self, payload, *, ok=True, status=200, url=""):
        self.ok = ok
        self.status = status
        self.url = url
        self.request = SimpleNamespace(method="POST")
        self._payload = payload

    async def json(self):
        return self._payload

    async def text(self):
        return json.dumps(self._payload, ensure_ascii=False)


class _Expect:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    @property
    async def value(self):
        return self._response


class _Locator:
    async def wait_for(self, **_kwargs):
        raise PlaywrightTimeoutError("hidden")

    async def is_visible(self):
        return False

    async def count(self):
        return 0

    def first(self):
        return self

    def nth(self, _index):
        return self

    def filter(self, **_kwargs):
        return self

    def locator(self, *_args, **_kwargs):
        return self

    def get_by_text(self, *_args, **_kwargs):
        return self

    def get_by_placeholder(self, *_args, **_kwargs):
        return self

    @property
    def last(self):
        return self


class _Page:
    def __init__(self, events, response, script=None):
        self.events = events
        self.url = "about:blank"
        self.response = response
        self.script = script or {}
        self.keyboard = self

    async def goto(self, url, **_kwargs):
        self.url = url
        self.events.append(("goto", url))
        if self.script.get("error") == "homepage":
            raise RuntimeError("homepage failed")
        if self.script.get("error") == "timeout":
            raise PlaywrightTimeoutError("detail timeout")
        if self.script.get("error") == "boom":
            raise RuntimeError("detail boom")

    async def evaluate(self, *_args, **_kwargs):
        return None

    def expect_response(self, *_args, **_kwargs):
        return _Expect(self.response)

    async def wait_for_selector(self, *_args, **_kwargs):
        return None

    def locator(self, *_args, **_kwargs):
        return _Locator()

    def get_by_text(self, *_args, **_kwargs):
        return _Locator()

    async def click(self, *_args, **_kwargs):
        raise PlaywrightTimeoutError("no dialog")

    async def press(self, *_args, **_kwargs):
        return None

    async def close(self):
        self.events.append(("close_page", self.url))


class _Context:
    def __init__(self, events, scripts, homepage_failures):
        self.events = events
        self.scripts = list(scripts)
        self.homepage_failures = homepage_failures
        self.scripts_added = []
        self._pages = 0

    async def add_init_script(self, script):
        self.scripts_added.append(script)
        self.events.append(("init_script", script))

    async def new_page(self):
        self.events.append(("new_page", self._pages))
        self._pages += 1
        if self._pages == 1:
            script = {}
            if self.homepage_failures[0] > 0:
                self.homepage_failures[0] -= 1
                script = {"error": "homepage"}
            return _Page(
                self.events,
                _Response(
                    {"data": {"resultList": []}},
                    url="https://h5api.m.goofish.com/h5/mtop.taobao.idlemtopsearch.pc.search/1.0/",
                ),
                script,
            )
        script = self.scripts.pop(0) if self.scripts else {"body": _success_detail()}
        return _Page(
            self.events,
            _Response(
                script.get("body", _success_detail()),
                ok=script.get("ok", True),
                status=script.get("status", 200),
                url="https://h5api.m.goofish.com/h5/mtop.taobao.idle.pc.detail/1.0/",
            ),
            script,
        )


class _Browser:
    def __init__(self, events, scripts, homepage_failures):
        self.events = events
        self.scripts = scripts
        self.homepage_failures = homepage_failures

    async def new_context(self, **_kwargs):
        self.events.append(("context", _kwargs))
        return _Context(self.events, self.scripts, self.homepage_failures)

    async def close(self):
        self.events.append(("close",))


class _Chromium:
    def __init__(self, events, scripts, homepage_failures):
        self.events = events
        self.scripts = scripts
        self.homepage_failures = homepage_failures

    async def launch(self, **kwargs):
        self.events.append(("launch", kwargs))
        return _Browser(self.events, self.scripts, self.homepage_failures)


class _Playwright:
    def __init__(self, events, scripts, homepage_failures):
        self.events = events
        self.scripts = scripts
        self.homepage_failures = homepage_failures
        self.chromium = _Chromium(events, scripts, homepage_failures)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


def _listings():
    return [
        {
            "商品标题": f"商品{index}的标题",
            "商品链接": f"https://www.goofish.com/item?id={index}",
            "当前售价": "100",
            "商品ID": str(index),
        }
        for index in (1, 2, 3)
    ]


def _success_detail():
    return {"ret": ["SUCCESS::调用成功"], "data": {"itemDO": {}, "sellerDO": {}}}


def _detail_gotos(events):
    urls = []
    for event in events:
        if len(event) >= 2 and event[0] == "goto" and "item?id=" in event[1]:
            urls.append(event[1])
    return urls


def _launches(events):
    return [event[1] for event in events if event[0] == "launch"]


def _assert_sandbox_kept(launch):
    args = launch["args"]
    for flag in FORBIDDEN_LAUNCH_FLAGS:
        assert flag not in args
    assert all("IsolateOrigins" not in arg and "site-per-process" not in arg for arg in args)
    assert "--disable-dev-shm-usage" in args
    assert any(arg.startswith("--crash-dumps-dir=") for arg in args)


def _assert_closed_after_last_detail(events):
    detail_at = [
        index
        for index, event in enumerate(events)
        if len(event) >= 2 and event[0] == "goto" and "item?id=" in event[1]
    ]
    close_at = [index for index, event in enumerate(events) if event[0] == "close"]
    assert close_at
    assert len(close_at) == len(_launches(events))
    if detail_at:
        assert max(detail_at) < close_at[-1]
    assert not any(event[0] == "goto" for event in events[close_at[-1] + 1 :])
    assert not any(event[0] == "init_script" for event in events)


def _run_scrape(
    monkeypatch,
    tmp_path,
    scripts,
    *,
    task_overrides=None,
    use_real_guard=False,
    homepage_failures=0,
    shutdown_sleep_seconds=None,
):
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    monkeypatch.setenv("ACCOUNT_ROTATION_RETRY_LIMIT", "3")
    monkeypatch.setenv("PROXY_ROTATION_RETRY_LIMIT", "3")

    state_path = tmp_path / "xianyu_state.json"
    state_path.write_text('{"cookies":[]}', encoding="utf-8")

    events = []
    advance_calls = []
    ai_calls = []
    notifications = []
    cleaned = []
    homepage_failure_budget = [homepage_failures]
    if use_real_guard:
        guard = FailureGuard(
            path=str(tmp_path / "task-failure-guard.json"),
            threshold=3,
            pause_seconds=3600,
            tz_name="Asia/Shanghai",
        )
    else:
        guard = _Guard()

    async def _parse(_payload, source):
        items = _listings()
        print(f"LOG: ({source}) 成功解析到 {len(items)} 条商品基础信息。")
        return items

    def _record(**_kwargs):
        return []

    async def _advance(**kwargs):
        advance_calls.append(kwargs.get("page_num"))
        return PageAdvanceResult(advanced=False, stop_reason="test-should-not-advance")

    async def _sleep(seconds):
        events.append(("sleep", seconds))

    async def _analyze(*_args, **_kwargs):
        ai_calls.append(True)
        raise AssertionError("AI analysis should not run after a validation block")

    async def _notify(*args, **_kwargs):
        notifications.append(args)

    def _cleanup(name):
        cleaned.append(name)

    def _no_input(*_args, **_kwargs):
        raise AssertionError("scrape blocked on input")

    def _randint(low, high):
        assert (low, high) == (3, 60)
        return shutdown_sleep_seconds

    monkeypatch.setattr(
        scraper,
        "async_playwright",
        lambda: _Playwright(events, scripts, homepage_failure_budget),
    )
    monkeypatch.setattr(scraper, "_parse_search_results_json", _parse)
    monkeypatch.setattr(scraper, "record_market_snapshots", _record)
    monkeypatch.setattr(scraper, "load_processed_link_keys", lambda _keyword: set())
    monkeypatch.setattr(scraper, "load_price_snapshots", lambda _keyword: [])
    monkeypatch.setattr(scraper, "cleanup_task_images", _cleanup)
    monkeypatch.setattr(scraper, "advance_search_page", _advance)
    monkeypatch.setattr(scraper, "FAILURE_GUARD", guard)
    monkeypatch.setattr(scraper, "get_ai_analysis", _analyze)
    monkeypatch.setattr(scraper, "send_ntfy_notification", _notify)
    monkeypatch.setattr(scraper.asyncio, "sleep", _sleep)
    monkeypatch.setattr("builtins.input", _no_input)
    if shutdown_sleep_seconds is not None:
        monkeypatch.setattr(scraper.random, "randint", _randint)

    task = {
        "task_name": "象印电饭煲-试抓",
        "keyword": "象印 电饭煲",
        "max_pages": 3,
        "personal_only": False,
        "free_shipping": False,
        "account_state_file": str(state_path),
        "account_strategy": "fixed",
    }
    if task_overrides:
        task.update(task_overrides)

    result = asyncio.run(scraper.scrape_xianyu(task))
    return {
        "result": result,
        "events": events,
        "guard": guard,
        "advance_calls": advance_calls,
        "ai_calls": ai_calls,
        "notifications": notifications,
        "cleaned": cleaned,
        "guard_path": tmp_path / "task-failure-guard.json",
    }


def _block(body, *, ok=True, status=200):
    return {"body": body, "ok": ok, "status": status}


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (
            {"ret": ["FAIL_SYS_USER_VALIDATE"], "data": {}},
            "FAIL_SYS_USER_VALIDATE",
        ),
        (
            {"ret": ["RGV587_ERROR::SM::哎哟喂,被挤爆啦,请稍后重试"], "data": {}},
            "RGV587_ERROR",
        ),
        (
            {
                "ret": ["SUCCESS::调用成功"],
                "data": {"url": "https://example.test/punish?x5secdata=abc"},
            },
            "FAIL_SYS_USER_VALIDATE",
        ),
    ],
)
def test_first_validation_block_stops_later_detail_requests(
    monkeypatch, tmp_path, capsys, body, reason
):
    outcome = _run_scrape(
        monkeypatch,
        tmp_path,
        [
            _block(body),
            _block(_success_detail()),
            _block(_success_detail()),
        ],
    )
    output = capsys.readouterr().out

    assert _detail_gotos(outcome["events"]) == ["https://www.goofish.com/item?id=1"]
    assert output.count("程序将终止") == 1
    assert f"检测到闲鱼反爬虫验证 ({reason})，程序将终止。" in output
    assert "检测到风控或验证触发" in output
    assert reason in output
    assert "处理商品详情时发生未知错误" not in output
    assert "爬取过程中发生未知错误" not in output
    assert "将尝试轮换" not in output
    assert "[页内进度 2/" not in output
    assert "开始处理第 2/" not in output
    assert outcome["advance_calls"] == []
    assert outcome["ai_calls"] == []
    assert outcome["result"] == 0
    assert len(_launches(outcome["events"])) == 1
    _assert_closed_after_last_detail(outcome["events"])
    _assert_sandbox_kept(_launches(outcome["events"])[0])
    assert outcome["guard"].failures
    assert outcome["guard"].failures[0][0] == "象印电饭煲-试抓"
    assert reason in outcome["guard"].failures[0][1]
    assert outcome["guard"].successes == []


def test_non_ok_validation_body_stops_before_the_next_detail(
    monkeypatch, tmp_path, capsys
):
    outcome = _run_scrape(
        monkeypatch,
        tmp_path,
        [
            _block(
                {"ret": ["FAIL_SYS_USER_VALIDATE::请稍后重试"], "data": {}},
                ok=False,
                status=419,
            ),
            _block(_success_detail()),
            _block(_success_detail()),
        ],
    )
    output = capsys.readouterr().out

    assert _detail_gotos(outcome["events"]) == ["https://www.goofish.com/item?id=1"]
    assert output.count("程序将终止") == 1
    assert "检测到风控或验证触发: FAIL_SYS_USER_VALIDATE" in output
    assert "处理商品详情时发生未知错误" not in output
    assert outcome["advance_calls"] == []
    assert outcome["guard"].successes == []


def test_ordinary_detail_error_still_fetches_until_the_validation_block(
    monkeypatch, tmp_path, capsys
):
    outcome = _run_scrape(
        monkeypatch,
        tmp_path,
        [
            {"error": "timeout"},
            _block({"ret": ["FAIL_SYS_USER_VALIDATE"], "data": {}}),
            _block(_success_detail()),
        ],
    )
    output = capsys.readouterr().out

    assert _detail_gotos(outcome["events"]) == [
        "https://www.goofish.com/item?id=1",
        "https://www.goofish.com/item?id=2",
    ]
    assert output.count("发现新商品，获取详情") == 2
    assert output.count("程序将终止") == 1
    assert output.find("[页内进度 2/") < output.find("程序将终止")
    assert "[页内进度 3/" not in output
    assert "开始处理第 2/" not in output
    assert "处理商品详情时发生未知错误" not in output
    assert "检测到风控或验证触发: FAIL_SYS_USER_VALIDATE" in output
    assert outcome["advance_calls"] == []
    assert outcome["ai_calls"] == []


def test_business_detail_errors_do_not_terminate_the_run(monkeypatch, tmp_path, capsys):
    offline = _block(
        {"ret": ["FAIL_BIZ_ITEM_OFFLINE::商品下架"], "data": {}},
        ok=False,
        status=200,
    )
    outcome = _run_scrape(monkeypatch, tmp_path, [offline, offline, offline])
    output = capsys.readouterr().out

    assert _detail_gotos(outcome["events"]) == [
        "https://www.goofish.com/item?id=1",
        "https://www.goofish.com/item?id=2",
        "https://www.goofish.com/item?id=3",
    ]
    assert "程序将终止" not in output
    assert "检测到风控或验证触发" not in output
    assert "将尝试轮换" not in output
    assert len(_launches(outcome["events"])) == 1
    _assert_closed_after_last_detail(outcome["events"])
    assert outcome["cleaned"] == ["象印电饭煲-试抓"]
    assert outcome["guard"].failures == []
    assert outcome["guard"].successes == ["象印电饭煲-试抓"]


def test_logged_terminate_stops_the_loop_even_if_abort_does_not_raise(
    monkeypatch, tmp_path, capsys
):
    async def _log_only(reason):
        print(f"检测到闲鱼反爬虫验证 ({reason})，程序将终止。")

    monkeypatch.setattr(scraper, "_abort_for_user_validation", _log_only)
    outcome = _run_scrape(
        monkeypatch,
        tmp_path,
        [
            _block({"ret": ["FAIL_SYS_USER_VALIDATE"], "data": {}}),
            _block(_success_detail()),
            _block(_success_detail()),
        ],
    )
    output = capsys.readouterr().out

    assert output.count("程序将终止") == 1
    assert _detail_gotos(outcome["events"]) == ["https://www.goofish.com/item?id=1"]
    assert "[页内进度 2/" not in output
    assert "开始处理第 2/" not in output
    assert outcome["advance_calls"] == []
    assert outcome["ai_calls"] == []
    _assert_closed_after_last_detail(outcome["events"])


def _rotation_task(tmp_path):
    account_dir = tmp_path / "accounts"
    account_dir.mkdir()
    for name in ("a.json", "b.json", "c.json"):
        (account_dir / name).write_text('{"cookies":[]}', encoding="utf-8")
    return {
        "account_strategy": "rotate",
        "account_state_file": None,
        "account_rotation": {
            "enabled": True,
            "mode": "on_failure",
            "retry_limit": 3,
            "state_dir": str(account_dir),
            "blacklist_ttl_sec": 300,
        },
        "proxy_rotation": {
            "enabled": True,
            "mode": "on_failure",
            "retry_limit": 3,
            "proxy_pool": "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003",
        },
    }


def test_validation_block_does_not_rotate_and_records_real_failure(
    monkeypatch, tmp_path, capsys
):
    outcome = _run_scrape(
        monkeypatch,
        tmp_path,
        [
            _block({"ret": ["FAIL_SYS_USER_VALIDATE"], "data": {}}),
            _block(_success_detail()),
            _block(_success_detail()),
        ],
        task_overrides=_rotation_task(tmp_path),
        use_real_guard=True,
        shutdown_sleep_seconds=17,
    )
    output = capsys.readouterr().out
    events = outcome["events"]
    launches = _launches(events)

    assert _detail_gotos(events) == ["https://www.goofish.com/item?id=1"]
    assert "https://www.goofish.com/item?id=2" not in _detail_gotos(events)
    assert len(launches) == 1
    assert output.count("账号轮换：使用登录状态") == 1
    assert output.count("IP 轮换：使用代理") == 1
    assert "将尝试轮换" not in output
    assert launches[0]["proxy"]["server"].startswith("http://127.0.0.1:800")
    _assert_sandbox_kept(launches[0])
    _assert_closed_after_last_detail(events)

    detail_at = next(
        index
        for index, event in enumerate(events)
        if len(event) >= 2 and event[1] == "https://www.goofish.com/item?id=1"
    )
    sleep_at = [
        index
        for index, event in enumerate(events)
        if event[0] == "sleep" and event[1] == 17
    ]
    close_at = next(index for index, event in enumerate(events) if event[0] == "close")
    assert len(sleep_at) == 1
    assert detail_at < sleep_at[0] < close_at
    assert "长时间休眠 (17 秒)" in output
    assert "长时间休眠结束，现在将安全退出。" in output
    assert output.count("程序将终止") == 1
    assert "检测到风控或验证触发: FAIL_SYS_USER_VALIDATE" in output
    assert "处理商品详情时发生未知错误" not in output
    assert "[FailureGuard] 任务 '象印电饭煲-试抓' 失败计数 1/3" in output
    assert outcome["notifications"] == []
    assert outcome["ai_calls"] == []
    assert outcome["advance_calls"] == []
    assert outcome["result"] == 0
    assert outcome["cleaned"] == ["象印电饭煲-试抓"]

    saved = json.loads(outcome["guard_path"].read_text(encoding="utf-8"))
    entry = saved["tasks"]["象印电饭煲-试抓"]
    assert entry["consecutive_failures"] == 1
    assert entry["last_failure_reason"] == "FAIL_SYS_USER_VALIDATE"
    assert entry["paused_until"] is None
    assert entry.get("last_success_at") is None


def test_ordinary_attempt_failure_still_rotates_account_and_proxy(
    monkeypatch, tmp_path, capsys
):
    outcome = _run_scrape(
        monkeypatch,
        tmp_path,
        [],
        task_overrides=_rotation_task(tmp_path),
        use_real_guard=True,
        homepage_failures=3,
    )
    output = capsys.readouterr().out
    launches = _launches(outcome["events"])
    proxies = [launch["proxy"]["server"] for launch in launches]

    assert _detail_gotos(outcome["events"]) == []
    assert len(launches) == 3
    assert len(set(proxies)) == 3
    assert output.count("将尝试轮换账号/IP 后重试...") == 2
    assert output.count("账号轮换：使用登录状态") == 3
    assert output.count("IP 轮换：使用代理") == 3
    assert "程序将终止" not in output
    assert "FAIL_SYS_USER_VALIDATE" not in output
    _assert_closed_after_last_detail(outcome["events"])
    for launch in launches:
        _assert_sandbox_kept(launch)

    saved = json.loads(outcome["guard_path"].read_text(encoding="utf-8"))
    entry = saved["tasks"]["象印电饭煲-试抓"]
    assert entry["consecutive_failures"] == 1
    assert entry["last_failure_reason"].startswith("RuntimeError: homepage failed")
    assert outcome["notifications"] == []
