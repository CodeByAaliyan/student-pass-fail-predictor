import asyncio
import time

# Normal synchronous function — blocks while sleeping
def sync_task():
    print("Task started")
    time.sleep(2)  # BLOCKS everything for 2 seconds
    print("Task done")


# Asynchronous function — yields control while waiting
async def async_task():
    print("Task started")
    await asyncio.sleep(2)  # Pauses this task, lets others run
    print("Task done")


# Running multiple async tasks together
async def main():
    await asyncio.gather(
        async_task(),  # Both tasks run concurrently
        async_task(),
    )


# Total time: ~2 seconds instead of ~4 seconds!
asyncio.run(main())