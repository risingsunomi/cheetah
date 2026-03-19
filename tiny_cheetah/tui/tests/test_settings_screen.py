import unittest

from textual.app import App
from textual.widgets import Static

from tiny_cheetah.tui.settings_screen import SettingsScreen


class _SettingsHost(App[None]):
    def compose(self):
        yield Static("host")


class TestSettingsScreen(unittest.IsolatedAsyncioTestCase):
    async def test_settings_screen_closes_on_escape(self) -> None:
        app = _SettingsHost()

        async with app.run_test() as pilot:
            screen = SettingsScreen()
            app.push_screen(screen)
            await pilot.pause()

            await pilot.press("escape")
            await pilot.pause()

            self.assertNotIsInstance(app.screen, SettingsScreen)


if __name__ == "__main__":
    unittest.main()
