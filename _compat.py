"""Python 3.9 compatibility shim for paddlex's GenAI (server-backend) client.

paddlex builds its rate-limiting `asyncio.Semaphore` in `GenAIClient.__init__`,
which runs on the main thread, but then awaits it on a *different* event loop
owned by a background thread (`_AsyncThreadManager`). On Python 3.9 a Semaphore
binds to the loop that exists at construction time, so this raises:

    RuntimeError: ... got Future <Future pending> attached to a different loop

Python 3.10 removed that eager binding, so the bug is 3.9-only. This shim
recreates the semaphore lazily, once per running loop. Import it before
constructing any server-backed pipeline. It is a no-op on 3.10+.
"""
import asyncio
import sys


def apply():
    if sys.version_info >= (3, 10):
        return False

    from paddlex.inference.models.common import genai

    def create_chat_completion(self, messages, *, return_future=False, **kwargs):
        async def _call():
            loop = asyncio.get_event_loop()
            sems = self.__dict__.setdefault("_sems_by_loop", {})
            sem = sems.get(loop)
            if sem is None:
                # Built inside the loop that will await it.
                sem = asyncio.Semaphore(self._max_concurrency)
                sems[loop] = sem
            async with sem:
                return await self._client.chat.completions.create(
                    model=self._model_name, messages=messages, **kwargs
                )

        return genai.run_async(_call(), return_future=return_future)

    genai.GenAIClient.create_chat_completion = create_chat_completion
    return True
