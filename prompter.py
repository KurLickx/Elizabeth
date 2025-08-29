import time
from constants import PATIENCE

class Prompter:
    def __init__(self, signals, llms, modules=None):
        self.signals = signals
        self.llms = llms
        if modules is None:
            self.modules = {}
        else:
            self.modules = modules
        self.system_ready = False
        self.timeSinceLastMessage = 0.0
    def prompt_now(self):
        # не промптить, если система не робит
        if not self.signals.stt_ready or not self.signals.tts_ready:
            return False
        # не промптить, если кто-то говорит
        if self.signals.human_speaking or self.signals.AI_thinking or self.signals.AI_speaking:
            return False
        # промптить, если кто-то что-то уже пизданул
        if self.signals.new_message:
            return True
        # промптить, если есть новые сообщения из твича
        if len(self.signals.recentTwitchMessages) > 0:
            return True
        # промптить, если много времени не пиздят
        if self.timeSinceLastMessage > PATIENCE:
            return True
    def chooseLLM(self):
        if "multimodal" in self.modules and self.modules["multimodal"].API.multimodal_now():
            return self.llms["image"]
        else:
            return self.llms["text"]
    def prompt_loop(self):
        print("Prompter loop started")
        while not self.signals.terminate:
            # начальная инициализация времени последнего сообщения
            if self.signals.last_message_time == 0.0 or (not self.signals.stt_ready or not self.signals.tts_ready):
                self.signals.last_message_time = time.time()
                self.timeSinceLastMessage = 0.0
            else:
                if not self.system_ready:
                    print("SYSTEM READY")
                    self.system_ready = True
            # просчет времени с пиздежа
            self.timeSinceLastMessage = time.time() - self.signals.last_message_time
            self.signals.sio_queue.put(("patience_update", {"crr_time": self.timeSinceLastMessage, "total_time": PATIENCE}))

            # Работа ллм
            if self.prompt_now():
                print("PROMPTING AI")
                llmWrapper = self.chooseLLM()
                llmWrapper.prompt()
                self.signals.last_message_time = time.time()
            # таймер чтоб не наебнуть систему
            time.sleep(0.1)
