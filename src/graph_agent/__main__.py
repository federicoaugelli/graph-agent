from __future__ import annotations

import argparse
import asyncio

from graph_agent.config import AppConfig, load_config
from graph_agent.core.service import AgentService
from graph_agent.logging import setup_logging


async def _cmd_cli(config: AppConfig) -> None:
    from graph_agent.core.sessions import SessionManager
    from graph_agent.transports.cli import run_cli

    service = AgentService(config)
    await service.setup()
    sessions = SessionManager(service, ttl_minutes=config.agent.session_ttl_minutes)
    await sessions.reset("cli", "local")
    try:
        await run_cli(service, sessions)
    finally:
        await service.shutdown()


async def _cmd_serve(config: AppConfig) -> None:
    import uvicorn

    from graph_agent.core.sessions import SessionManager
    from graph_agent.transports.http.app import create_app

    service = AgentService(config)
    await service.setup()
    sessions = SessionManager(service, ttl_minutes=config.agent.session_ttl_minutes)

    realtime = None
    if config.channels.realtime.enabled:
        from graph_agent.transports.realtime import RealtimeBridge

        realtime = RealtimeBridge(service, config.channels.realtime)

    app = create_app(service, config, sessions, realtime)
    http = config.channels.http
    server = uvicorn.Server(uvicorn.Config(app, host=http.host, port=http.port, log_level="info"))

    tasks = [asyncio.create_task(server.serve())]
    telegram = None
    if config.channels.telegram.enabled:
        from graph_agent.transports.telegram import TelegramBot

        telegram = TelegramBot(service, config.channels.telegram, sessions)
        tasks.append(asyncio.create_task(telegram.start()))

    scheduler = None
    if config.scheduler.enabled:
        from graph_agent.scheduler.cron import CronScheduler

        scheduler = CronScheduler(service, config.scheduler)
        await scheduler.start()
        from graph_agent.tools.schedule import ScheduleTool

        service.registry.register(ScheduleTool(scheduler))
        if telegram is not None and config.channels.telegram.allowed_user_ids:
            from aiogram.exceptions import TelegramBadRequest
            from aiogram.types import FSInputFile

            from graph_agent.transports.telegram.bot import send_telegram_html
            from graph_agent.transports.telegram.formatting import (
                html_to_plain,
                markdown_to_telegram_html,
            )

            chat_id = config.channels.telegram.allowed_user_ids[0]
            bot = telegram.bot

            class TelegramSink:
                async def send(self, text: str) -> None:
                    await send_telegram_html(bot, chat_id, text)

                async def send_file(self, path: str, caption: str | None = None) -> None:
                    if caption is None:
                        await bot.send_document(chat_id=chat_id, document=FSInputFile(path))
                        return
                    formatted = markdown_to_telegram_html(caption)
                    try:
                        await bot.send_document(
                            chat_id=chat_id,
                            document=FSInputFile(path),
                            caption=formatted,
                            parse_mode="HTML",
                        )
                    except TelegramBadRequest:
                        await bot.send_document(
                            chat_id=chat_id,
                            document=FSInputFile(path),
                            caption=html_to_plain(formatted),
                        )

            scheduler.register_sink("telegram", TelegramSink())

    try:
        await asyncio.gather(*tasks)
    finally:
        if scheduler is not None:
            await scheduler.shutdown()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        if telegram is not None:
            await telegram.stop()
        await service.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser(prog="graph-agent")
    parser.add_argument("--config", default="config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("cli", help="interactive terminal REPL")
    sub.add_parser("serve", help="start server transports (HTTP, Telegram, ...)")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging(config.logging)

    if args.command == "cli":
        asyncio.run(_cmd_cli(config))
    elif args.command == "serve":
        asyncio.run(_cmd_serve(config))


if __name__ == "__main__":
    main()
