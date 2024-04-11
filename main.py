import asyncio
import logging

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.fsm.storage.memory import MemoryStorage

from HR import SalesGPT, llm

bot_token = '6680547488:AAGw_y0ncoSJf7UqFUvb36Yggg9PQI0BjNY'

sales_agent = None

async def main():

    storage = MemoryStorage()
    dp = Dispatcher(storage=storage)
    bot = Bot(bot_token, parse_mode=None)
    logging.basicConfig(level=logging.INFO)

    @dp.message(Command(commands=["start"]))
    async def repl(message):
        global sales_agent
        sales_agent = SalesGPT.from_llm(llm, verbose=False)
        sales_agent.seed_agent()
        await message.answer('Генерируется ответ♻️')  # Даём понять пользователю, что бот работает
        ai_message = sales_agent.ai_step()
        await message.answer(ai_message)
        await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id + 1)

    @dp.message(F.text)
    async def repl(message):
        if sales_agent is None:
            await message.answer('Используйте команду /start')
        else:
            human_message = message.text
            if human_message:
                sales_agent.human_step(human_message)
                sales_agent.analyse_stage()
            await message.answer('Генерируется ответ♻️')
            ai_message = sales_agent.ai_step()
            await message.answer(ai_message)
            await bot.delete_message(chat_id=message.chat.id, message_id=message.message_id + 1)

    @dp.message(~F.text)
    async def empty(message):
        await message.answer('Бот принимает только текст')

    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())