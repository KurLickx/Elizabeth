import copy
import requests
import sseclient
import json
import time
from dotenv import load_dotenv
from constants import *
from modules.injection import Injection

class AbstractLLMWrapper:
    def __init__(self, signals, tts, llmState, modules=None):
        self.signals = signals
        self.llmState = llmState
        self.tts = tts
        self.API = self.API(self)
        if modules is None:
            self.modules = {}
        else:
            self.modules = modules
        self.headers = {"Content-Type": "application/json"}
        load_dotenv()
        self.SYSTEM_PROMPT = None
        self.LLM_ENDPOINT = None
        self.CONTEXT_SIZE = None
        self.tokenizer = None
    def is_filtered(self, text):
        if any(bad_word.lower() in text.lower().split() for bad_word in self.llmState.blacklist):
            return True
        else:
            return False
    def assemble_injections(self, injections=None):
        if injections is None:
            injections = []
        for module in self.modules.values():
            injections.append(module.get_prompt_injection())
        for module in self.modules.values():
            module.cleanup()
        injections = sorted(injections, key=lambda x: x.priority)
        prompt = ""
        for injection in injections:
            prompt += injection.text
        return prompt
    def generate_prompt(self):
        messages = copy.deepcopy(self.signals.history)
        for message in messages:
            if message["role"] == "user" and message["content"] != "":
                message["content"] = HOST_NAME + ": " + message["content"] + "\n"
            elif message["role"] == "assistant" and message["content"] != "":
                message["content"] = AI_NAME + ": " + message["content"] + "\n"
        while True:
            chat_section = ""
            for message in messages:
                chat_section += message["content"]
            generation_prompt = AI_NAME + ": "
            base_injections = [Injection(self.SYSTEM_PROMPT, 10), Injection(chat_section, 100)]
            full_prompt = self.assemble_injections(base_injections) + generation_prompt
            wrapper = [{"role": "user", "content": full_prompt}]
            prompt_tokens = len(self.tokenizer.apply_chat_template(wrapper, tokenize=True, return_tensors="pt")[0])
            if prompt_tokens < 0.9 * self.CONTEXT_SIZE:
                self.signals.sio_queue.put(("full_prompt", full_prompt))
                return full_prompt
            else:
                if len(messages) < 1:
                    raise RuntimeError("Промпт слишком длинный, не могу сократить!")
                messages.pop(0)
                print("Промпт слишком длинный, удаление старых сообщений...")
    def prepare_payload(self):
        raise NotImplementedError("Must implement prepare_payload in child classes")
    def prompt(self):
        if not self.llmState.enabled:
            return
        self.signals.AI_thinking = True
        self.signals.new_message = False
        self.signals.sio_queue.put(("reset_next_message", None))
        data = self.prepare_payload()
        stream_response = requests.post(self.LLM_ENDPOINT + "/v1/chat/completions", headers=self.headers, json=data,
                                        verify=False, stream=True)
        response_stream = sseclient.SSEClient(stream_response)
        AI_message = ''
        for event in response_stream.events():
            if self.llmState.next_cancelled:
                continue
            payload = json.loads(event.data)
            chunk = payload['choices'][0]['delta']['content']
            AI_message += chunk
            self.signals.sio_queue.put(("next_chunk", chunk))
        if self.llmState.next_cancelled:
            self.llmState.next_cancelled = False
            self.signals.sio_queue.put(("reset_next_message", None))
            self.signals.AI_thinking = False
            return
        print("AI OUTPUT: " + AI_message)
        self.signals.last_message_time = time.time()
        self.signals.AI_speaking = True
        self.signals.AI_thinking = False
        if self.is_filtered(AI_message):
            AI_message = "Filtered."
            self.signals.sio_queue.put(("reset_next_message", None))
            self.signals.sio_queue.put(("next_chunk", "Filtered."))
        self.signals.history.append({"role": "assistant", "content": AI_message})
        self.tts.play(AI_message)
    class API:
        def __init__(self, outer):
            self.outer = outer
        def get_blacklist(self):
            return self.outer.llmState.blacklist
        def set_blacklist(self, new_blacklist):
            self.outer.llmState.blacklist = new_blacklist
            with open('blacklist.txt', 'w') as file:
                for word in new_blacklist:
                    file.write(word + "\n")
            self.outer.signals.sio_queue.put(('get_blacklist', new_blacklist))
        def set_LLM_status(self, status):
            self.outer.llmState.enabled = status
            if status:
                self.outer.signals.AI_thinking = False
            self.outer.signals.sio_queue.put(('LLM_status', status))
        def get_LLM_status(self):
            return self.outer.llmState.enabled
        def cancel_next(self):
            self.outer.llmState.next_cancelled = True
            requests.post(self.outer.LLM_ENDPOINT + "/v1/internal/stop-generation", headers={"Content-Type": "application/json"})
