import os
import asyncio
import logging
from typing import List, Dict, Any, Optional
from groq import AsyncGroq
import groq

# Set up logging
logger = logging.getLogger("api_rotator")
logger.setLevel(logging.INFO)

class RotatingCompletions:
    def __init__(self, parent: RotatingAsyncGroq):
        self.parent = parent

    async def create(self, *args, **kwargs) -> Any:
        num_keys = len(self.parent.keys)
        if num_keys == 0:
            logger.warning("[Rotator] No API keys configured in environment! Falling back to empty client.")
            client = self.parent.get_default_client()
            return await client.chat.completions.create(*args, **kwargs)

        last_exception = None
        # Try at most num_keys + 1 times (so we try every key once, plus one retry if needed)
        for attempt in range(num_keys + 1):
            client = await self.parent.get_next_client()
            # Calculate the index of the key we just retrieved
            key_index = self.parent.index - 1
            if key_index < 0:
                key_index = num_keys - 1
                
            key_preview = client.api_key[:10] + "..." if client.api_key else "None"
            print(f"[Rotator] Attempting Groq request (Attempt {attempt+1}/{num_keys+1}) using Key Index {key_index} ({key_preview})")
            
            try:
                # Call the actual create method on the selected AsyncGroq client
                return await client.chat.completions.create(*args, **kwargs)
            except groq.RateLimitError as exc:
                print(f"[Rotator] RateLimitError on Key Index {key_index} ({key_preview}): {exc}")
                last_exception = exc
                # Rotate and try immediately on rate limit after a small pause
                await asyncio.sleep(0.5)
            except Exception as exc:
                print(f"[Rotator] Exception on Key Index {key_index} ({key_preview}): {exc}")
                last_exception = exc
                # Rotate and try next key
                await asyncio.sleep(0.5)

        # If all keys failed, raise the last exception
        if last_exception:
            raise last_exception
        raise RuntimeError("All configured Groq API keys failed to process the request.")

class RotatingChat:
    def __init__(self, parent: RotatingAsyncGroq):
        self.completions = RotatingCompletions(parent)

class RotatingAsyncGroq:
    """
    Custom wrapper that emulates AsyncGroq but rotates API keys
    and transparently handles failover across multiple keys.
    """
    def __init__(self):
        # 1. Load keys from GROQ_API_KEYS (comma separated)
        raw_keys = os.getenv("GROQ_API_KEYS", "")
        keys = [k.strip() for k in raw_keys.split(",") if k.strip()]
        
        # 2. Fall back to standard GROQ_API_KEY if list is empty
        if not keys:
            single_key = os.getenv("GROQ_API_KEY", "")
            if single_key:
                keys = [single_key]
                
        self.keys = keys
        self.clients = [AsyncGroq(api_key=k) for k in keys]
        self.index = 0
        self._lock = asyncio.Lock()
        
        # Build mimic structure
        self.chat = RotatingChat(self)
        print(f"[Rotator] Initialized with {len(keys)} Groq API keys.")

    def get_default_client(self) -> AsyncGroq:
        return AsyncGroq()

    async def get_next_client(self) -> AsyncGroq:
        if not self.clients:
            return AsyncGroq()
        async with self._lock:
            client = self.clients[self.index]
            self.index = (self.index + 1) % len(self.clients)
            return client
