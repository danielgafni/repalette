import discord
from repalette.constants import DISCORD_BOT_TOKEN


async def notify_discord(channel_id, message):
    client = discord.Client()

    async def __send_message():
        await client.wait_until_ready()
        await client.get_channel(channel_id).send(message)
        await client.close()

    client.loop.create_task(__send_message())
    await client.start(DISCORD_BOT_TOKEN)
