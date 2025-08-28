import asyncio
from modules.injection import Injection
class Module:
    def __init__(self, signals, enabled=True):
        self.signals = signals
        self.enabled = enabled
        self.prompt_injection = Injection("", -1)
    def init_event_loop(self):
        asyncio.run(self.run())
    def get_prompt_injection(self):
        return self.prompt_injection
    def cleanup(self):
        pass
    async def run(self):
        pass
