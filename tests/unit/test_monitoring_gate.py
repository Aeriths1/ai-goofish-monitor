import asyncio
import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api import dependencies as deps
from src.api.routes import settings as settings_routes
from src.infrastructure.persistence.sqlite_connection import sqlite_connection
from src.infrastructure.persistence.sqlite_task_repository import SqliteTaskRepository
from src.monitoring_gate import monitoring_block_reason, monitoring_enabled
from src.services.process_service import ProcessService
from src.services.scheduler_service import SchedulerService
from src.services.task_service import TaskService


FORBIDDEN_LAUNCH_FLAGS = (
    "--disable-web-security",
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-setuid-sandbox",
)


class _IdleProcessService:
    def __init__(self):
        self.processes = {}


class _Guard:
    threshold = 3

    def should_skip_start(self, *_args, **_kwargs):
        return SimpleNamespace(
            skip=False,
            should_notify=False,
            reason="not_paused",
            paused_until=None,
            consecutive_failures=0,
        )

    def record_failure(self, *_args, **_kwargs):
        return {
            "should_notify": False,
            "consecutive_failures": 1,
            "paused_until": None,
        }

    def record_success(self, *_args, **_kwargs):
        return None


class _Page:
    def __init__(self, events):
        self.events = events
        self.url = "about:blank"

    async def goto(self, url, **_kwargs):
        self.url = url
        self.events.append(("goto", url))
        raise RuntimeError("offline-boundary")


class _Context:
    def __init__(self, events):
        self.events = events
        self.scripts = []

    async def add_init_script(self, script):
        self.scripts.append(script)
        self.events.append(("init_script", script))

    async def new_page(self):
        self.events.append(("new_page",))
        return _Page(self.events)


class _Browser:
    def __init__(self, events):
        self.events = events
        self.context = None

    async def new_context(self, **kwargs):
        self.events.append(("context", kwargs))
        self.context = _Context(self.events)
        return self.context

    async def close(self):
        self.events.append(("close",))


class _Chromium:
    def __init__(self, events):
        self.events = events

    async def launch(self, **kwargs):
        self.events.append(("launch", kwargs))
        return _Browser(self.events)


class _Playwright:
    def __init__(self, events):
        self.events = events
        self.chromium = _Chromium(events)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False


def _task_rows(db_path: str):
    with sqlite_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT task_name, enabled, keyword, description, cron, is_running
            FROM tasks
            ORDER BY id
            """
        ).fetchall()
    return [dict(row) for row in rows]


def test_monitoring_enabled_is_opt_in(monkeypatch):
    monkeypatch.delenv("MONITORING_ENABLED", raising=False)
    assert monitoring_enabled() is False

    for value in ("", "false", "0", "no", "off", " FALSE "):
        monkeypatch.setenv("MONITORING_ENABLED", value)
        assert monitoring_enabled() is False

    for value in ("1", "true", "YES", "on"):
        monkeypatch.setenv("MONITORING_ENABLED", value)
        assert monitoring_enabled() is True


def test_prep_restart_keeps_user_task_and_does_not_schedule(tmp_path, monkeypatch):
    monkeypatch.delenv("MONITORING_ENABLED", raising=False)
    db_path = str(tmp_path / "app.sqlite3")
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            [
                {
                    "task_name": "已有任务",
                    "enabled": True,
                    "keyword": "keep-me",
                    "description": "user data",
                    "max_pages": 2,
                    "personal_only": False,
                    "cron": "0 0 1 1 *",
                    "ai_prompt_base_file": "prompts/base_prompt.txt",
                    "ai_prompt_criteria_file": "prompts/criteria.txt",
                    "is_running": True,
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    repository = SqliteTaskRepository(
        db_path=db_path,
        legacy_config_file=str(config_path),
    )
    spawned = []

    async def _run():
        tasks = await repository.find_all()
        assert tasks[0].enabled is True
        assert tasks[0].is_running is True
        before = _task_rows(db_path)

        service = TaskService(repository)
        for task in tasks:
            if task.is_running:
                await service.update_task_status(task.id, False)

        scheduler = SchedulerService(ProcessService())

        async def _boom(*args, **_kwargs):
            spawned.append(args)
            raise AssertionError("scheduler spawned a scraper")

        scheduler.process_service.start_task = _boom
        await scheduler.reload_jobs(await repository.find_all())

        assert scheduler.scheduler.get_jobs() == []
        kept = await repository.find_all()
        assert kept[0].enabled is True
        assert kept[0].keyword == "keep-me"
        assert kept[0].description == "user data"
        assert kept[0].cron == "0 0 1 1 *"
        assert kept[0].is_running is False
        after = _task_rows(db_path)
        assert after[0]["enabled"] == before[0]["enabled"] == 1
        assert after[0]["keyword"] == before[0]["keyword"]
        assert after[0]["description"] == before[0]["description"]
        assert after[0]["cron"] == before[0]["cron"]

    asyncio.run(_run())
    assert spawned == []


def test_example_config_imports_without_enabling_or_scheduling(tmp_path, monkeypatch):
    monkeypatch.delenv("MONITORING_ENABLED", raising=False)
    example = json.loads(Path("config.json.example").read_text(encoding="utf-8"))
    assert example[0]["enabled"] is False
    assert example[0]["keyword"] == "苹果watch S10"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(example, ensure_ascii=False), encoding="utf-8")
    db_path = str(tmp_path / "example.sqlite3")
    repository = SqliteTaskRepository(
        db_path=db_path,
        legacy_config_file=str(config_path),
    )

    async def _run():
        tasks = await repository.find_all()
        assert len(tasks) == 1
        assert tasks[0].enabled is False
        assert tasks[0].keyword == "苹果watch S10"
        assert tasks[0].description.startswith("九成新")
        scheduler = SchedulerService(ProcessService())
        await scheduler.reload_jobs(tasks)
        assert scheduler.scheduler.get_jobs() == []
        again = await repository.find_all()
        assert again[0].task_name == "苹果watch S10"
        assert again[0].enabled is False

    asyncio.run(_run())
    assert _task_rows(db_path)[0]["keyword"] == "苹果watch S10"


def test_process_start_is_idle_and_silent_in_prep_mode(monkeypatch):
    monkeypatch.delenv("MONITORING_ENABLED", raising=False)
    notifications = []

    async def _notify(*args, **_kwargs):
        notifications.append(args)
        raise AssertionError("notification sent")

    async def _spawn(*_args, **_kwargs):
        raise AssertionError("browser process spawned")

    monkeypatch.setattr(
        "src.services.process_service.send_ntfy_notification",
        _notify,
    )
    monkeypatch.setattr(asyncio, "create_subprocess_exec", _spawn)
    service = ProcessService()

    async def _run():
        started = await service.start_task(7, "示例任务")
        assert started is False
        assert service.is_running(7) is False

    asyncio.run(_run())
    assert notifications == []


def test_scrape_prep_mode_stops_before_browser_storage_ai_or_notification(monkeypatch):
    monkeypatch.delenv("MONITORING_ENABLED", raising=False)
    import src.scraper as scraper

    def _explode(*_args, **_kwargs):
        raise AssertionError("prep mode reached a live side effect")

    monkeypatch.setattr(scraper, "async_playwright", _explode)
    monkeypatch.setattr(scraper, "send_ntfy_notification", _explode)
    monkeypatch.setattr(scraper, "get_ai_analysis", _explode)
    monkeypatch.setattr(scraper, "load_processed_link_keys", _explode)
    monkeypatch.setattr(scraper, "load_price_snapshots", _explode)

    async def _run():
        result = await scraper.scrape_xianyu(
            {"task_name": "示例任务", "keyword": "do-not-fetch"}
        )
        assert result == 0

    asyncio.run(_run())


def test_armed_scrape_keeps_browser_security_and_stops_offline(tmp_path, monkeypatch):
    monkeypatch.setenv("MONITORING_ENABLED", "true")
    monkeypatch.setenv("APP_DATABASE_FILE", str(tmp_path / "app.sqlite3"))
    monkeypatch.setenv("ACCOUNT_ROTATION_RETRY_LIMIT", "1")
    monkeypatch.setenv("PROXY_ROTATION_RETRY_LIMIT", "1")
    import src.scraper as scraper

    state_path = tmp_path / "state.json"
    state_path.write_text('{"cookies":[]}', encoding="utf-8")
    monkeypatch.setattr(scraper, "STATE_FILE", str(state_path))
    monkeypatch.setattr(scraper, "FAILURE_GUARD", _Guard())

    events = []
    notifications = []
    ai_calls = []

    def _async_playwright():
        return _Playwright(events)

    async def _notify(*args, **_kwargs):
        notifications.append(args)

    async def _analyze(*_args, **_kwargs):
        ai_calls.append(True)
        raise AssertionError("AI called")

    async def _sleep(_seconds):
        return None

    monkeypatch.setattr(scraper, "async_playwright", _async_playwright)
    monkeypatch.setattr(scraper, "send_ntfy_notification", _notify)
    monkeypatch.setattr(scraper, "get_ai_analysis", _analyze)
    monkeypatch.setattr(scraper.asyncio, "sleep", _sleep)

    async def _run():
        result = await scraper.scrape_xianyu(
            {
                "task_name": "offline",
                "keyword": "offline-keyword",
                "account_state_file": str(state_path),
                "account_strategy": "fixed",
                "max_pages": 1,
            }
        )
        assert result == 0

    asyncio.run(_run())

    launch = next(item for item in events if item[0] == "launch")
    args = launch[1]["args"]
    for flag in FORBIDDEN_LAUNCH_FLAGS:
        assert flag not in args
    assert all("IsolateOrigins" not in arg and "site-per-process" not in arg for arg in args)
    assert not any(item[0] == "init_script" for item in events)
    gotos = [item[1] for item in events if item[0] == "goto"]
    assert gotos == ["https://www.goofish.com/"]
    assert notifications == []
    assert ai_calls == []


def test_cli_prep_mode_does_not_run_example_or_enabled_tasks(tmp_path, monkeypatch):
    monkeypatch.delenv("MONITORING_ENABLED", raising=False)
    example = json.loads(Path("config.json.example").read_text(encoding="utf-8"))
    example.append(
        {
            **example[0],
            "task_name": "已启用任务",
            "enabled": True,
            "cron": "0 0 1 1 *",
        }
    )
    config_path = tmp_path / "config.json"
    original = json.dumps(example, ensure_ascii=False)
    config_path.write_text(original, encoding="utf-8")

    sys.modules.pop("spider_v2", None)
    spider_v2 = importlib.import_module("spider_v2")
    called = []

    async def _scrape(*_args, **_kwargs):
        called.append(True)
        raise AssertionError("scrape started")

    monkeypatch.setattr(spider_v2, "scrape_xianyu", _scrape)
    monkeypatch.setattr(
        sys,
        "argv",
        ["spider_v2.py", "--config", str(config_path)],
    )

    asyncio.run(spider_v2.main())

    assert called == []
    assert config_path.read_text(encoding="utf-8") == original
    assert monitoring_block_reason()


def test_status_endpoint_reports_prep_mode(monkeypatch):
    monkeypatch.delenv("MONITORING_ENABLED", raising=False)
    app = FastAPI()
    app.include_router(settings_routes.router)
    app.dependency_overrides[deps.get_process_service] = _IdleProcessService
    client = TestClient(app)

    response = client.get("/api/settings/status")

    assert response.status_code == 200
    body = response.json()
    assert body["monitoring_enabled"] is False
    assert body["monitoring_mode"] == "prep"
    assert body["scraper_running"] is False


def test_scraper_source_has_no_webdriver_stealth_injection():
    source = Path("src/scraper.py").read_text(encoding="utf-8")
    assert "add_init_script" not in source
    assert "AutomationControlled" not in source
    assert "--disable-web-security" not in source
    assert "绕过检测" not in source
