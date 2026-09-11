"""The same round trips with asyncio's streams, as the standard library writes an echo server."""

import asyncio
import time

ROUNDS = 20000


async def serve(reader, writer):
    total = 0
    for _ in range(ROUNDS):
        data = await reader.readexactly(8)
        writer.write(data)
        await writer.drain()
        total += len(data)
    writer.close()
    await writer.wait_closed()
    return total


async def main():
    served = asyncio.get_running_loop().create_future()

    async def on_client(reader, writer):
        served.set_result(await serve(reader, writer))

    server = await asyncio.start_server(on_client, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    got = 0
    for _ in range(ROUNDS):
        writer.write(b"abcdefgh")
        await writer.drain()
        got += len(await reader.readexactly(8))
    writer.close()
    await writer.wait_closed()
    sent = await served
    server.close()
    await server.wait_closed()
    return got + sent


started = time.perf_counter()
answer = asyncio.run(main())
took = (time.perf_counter() - started) * 1000.0
print(f"# {ROUNDS} round trips: {took:.2f} ms")
print(answer)
