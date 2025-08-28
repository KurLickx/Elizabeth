import signal
import sys
import time
import threading
import asyncio
from signals import Signals
from prompter import Prompter
from llmWrappers.llmState import LLMState
from llmWrappers.textLLMWrapper import TextLLMWrapper
from llmWrappers.imageLLMWrapper import ImageLLMWrapper
from stt import STT
from tts import TTS
from modules.twitchClient import TwitchClient
from modules.audioPlayer import AudioPlayer
from modules.vtubeStudio import VtubeStudio
from modules.multimodal import MultiModal
from modules.customPrompt import CustomPrompt
from modules.memory import Memory
from socketioServer import SocketIOServer

async def main():
    print("Starting Project...")
    def signal_handler(sig, frame):
        print('Received CTRL + C, attempting to gracefully exit. Close all dashboard windows to speed up shutdown.')
        signals.terminate = True
        stt.API.shutdown()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    signals = Signals()
    modules = {}
    module_threads = {}
    stt = STT(signals)
    tts = TTS(signals)
    llmState = LLMState()
    llms = {
        "text": TextLLMWrapper(signals, tts, llmState, modules),
        "image": ImageLLMWrapper(signals, tts, llmState, modules)
    }
    prompter = Prompter(signals, llms, modules)
    modules['twitch'] = TwitchClient(signals, enabled=False)
    modules['audio_player'] = AudioPlayer(signals, enabled=True)
    modules['vtube_studio'] = VtubeStudio(signals, enabled=False) #Пока без шмота
    modules['multimodal'] = MultiModal(signals, enabled=False) #Пока без мултимодала
    modules['custom_prompt'] = CustomPrompt(signals, enabled=True)
    modules['memory'] = Memory(signals, enabled=True)
    sio = SocketIOServer(signals, stt, tts, llms["text"], prompter, modules=modules)
    prompter_thread = threading.Thread(target=prompter.prompt_loop, daemon=True)
    stt_thread = threading.Thread(target=stt.listen_loop, daemon=True)
    sio_thread = threading.Thread(target=sio.start_server, daemon=True)
    sio_thread.start()
    prompter_thread.start()
    stt_thread.start()
    for name, module in modules.items():
        module_thread = threading.Thread(target=module.init_event_loop, daemon=True)
        module_threads[name] = module_thread
        module_thread.start()
    while not signals.terminate:
        time.sleep(0.1)
    print("ГГ ВП ======================")
    for module_thread in module_threads.values():
        module_thread.join()
    sio_thread.join()
    print("Сиошка убита ======================")
    prompter_thread.join()
    print("Пропты не робят ======================")
    stt_thread.join()
    print("Войс не слышен ======================")
    print("Все потоки закрыты, ну нахуй")
    sys.exit(0)

if __name__ == '__main__':
    asyncio.run(main())
