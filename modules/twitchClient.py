from twitchAPI.twitch import Twitch
from twitchAPI.oauth import UserAuthenticator
from twitchAPI.type import AuthScope, ChatEvent
from twitchAPI.chat import Chat, EventData, ChatMessage, ChatSub, ChatCommand
import os
import asyncio
from dotenv import load_dotenv
from constants import TWITCH_CHANNEL, TWITCH_MAX_MESSAGE_LENGTH
from modules.module import Module
class TwitchClient(Module):
    def __init__(self, signals, enabled=True):
        super().__init__(signals, enabled)
        self.chat = None
        self.twitch = None
        self.API = self.API(self)
        self.prompt_injection.priority = 150
    def get_prompt_injection(self):
        if len(self.signals.recentTwitchMessages) > 0:
            output = "\nСообщений нет :( \n"
            for message in self.signals.recentTwitchMessages:
                output += message + "\n"
            output += "Выбери савмое пиздатое соббщение и ответь на него.\n"
            self.prompt_injection.text = output
        else:
            self.prompt_injection.text = ""
        return self.prompt_injection
    def cleanup(self):
        self.signals.recentTwitchMessages = []
    async def run(self):
        load_dotenv()
        APP_ID = os.getenv("TWITCH_APP_ID") #а тута я не ебу что это но оно обязательно надо
        APP_SECRET = os.getenv("TWITCH_SECRET") #тута свой ключ (в энве я долбоёб)
        USER_SCOPE = [AuthScope.CHAT_READ, AuthScope.CHAT_EDIT]
        async def on_ready(ready_event: EventData):
            print('TWITCH: Bot is ready for work, joining channels')
            await ready_event.chat.join_room(TWITCH_CHANNEL)
        async def on_message(msg: ChatMessage):
            if not self.enabled:
                return
            if len(msg.text) > TWITCH_MAX_MESSAGE_LENGTH:
                return
            print(f'in {msg.room.name}, {msg.user.name} said: {msg.text}')
            if len(self.signals.recentTwitchMessages) > 10:
                self.signals.recentTwitchMessages.pop(0)
            self.signals.recentTwitchMessages.append(f"{msg.user.name} : {msg.text}")
            self.signals.recentTwitchMessages = self.signals.recentTwitchMessages
        async def on_sub(sub: ChatSub):
            print(f'Новая подписка(атписька) {sub.room.name}:\\n'
                  f'  Type: {sub.sub_plan}\\n'
                  f'  Message: {sub.sub_message}')
        async def test_command(cmd: ChatCommand):
            if len(cmd.parameter) == 0:
                await cmd.reply('ТЫ НЕ СКАЗАЛ ЧТО ГОВОРИТЬ ЫЫЫЫ')
            else:
                await cmd.reply(f'{cmd.user.name}: {cmd.parameter}')
        if not self.enabled:
            return
        twitch = await Twitch(APP_ID, APP_SECRET)
        auth = UserAuthenticator(twitch, USER_SCOPE)
        token, refresh_token = await auth.authenticate()
        await twitch.set_user_authentication(token, USER_SCOPE, refresh_token)
        chat = await Chat(twitch)
        self.twitch = twitch
        self.chat = chat
        chat.register_event(ChatEvent.READY, on_ready)
        chat.register_event(ChatEvent.MESSAGE, on_message)
        chat.register_event(ChatEvent.SUB, on_sub)
        chat.register_command('reply', test_command)
        chat.start()
        while True:
            if self.signals.terminate:
                self.chat.stop()
                await self.twitch.close()
                return
            await asyncio.sleep(0.1)
    class API:
        def __init__(self, outer):
            self.outer = outer
        def set_twitch_status(self, status):
            self.outer.enabled = status
            if not status:
                self.outer.signals.recentTwitchMessages = []
            self.outer.signals.sio_queue.put(('twitch_status', status))
        def get_twitch_status(self):
            return self.outer.enabled
