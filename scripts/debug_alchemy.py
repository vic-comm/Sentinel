# Run this one-off debug script first
# scripts/debug_alchemy.py

import asyncio
import aiohttp
import os
from dotenv import load_dotenv

load_dotenv()
ALCHEMY_KEY = os.getenv("ALCHEMY_API_KEY")
URL = f"https://eth-mainnet.g.alchemy.com/v2/{ALCHEMY_KEY}"

TORNADO = "0x47CE0C6eD5B0Ce3d3A51fdb1C52DC66a7c3c2936"

async def debug():
    # Test 1: raw API response
    payload = {
        "jsonrpc": "2.0",
        "method": "alchemy_getAssetTransfers",
        "params": [{
            "fromBlock": "0x0",
            "toBlock":   "latest",
            "toAddress": TORNADO,
            "category":  ["external"],
            "withMetadata": True,
            "maxCount":  hex(5),
        }],
        "id": 1,
    }

    async with aiohttp.ClientSession() as session:
        async with session.post(URL, json=payload,
                                timeout=aiohttp.ClientTimeout(total=30)) as r:
            status = r.status
            data   = await r.json()

    print(f"HTTP status: {status}")
    print(f"Keys in response: {list(data.keys())}")

    if "error" in data:
        print(f"API ERROR: {data['error']}")
    elif "result" in data:
        result = data["result"]
        print(f"Result keys: {list(result.keys())}")
        transfers = result.get("transfers", [])
        print(f"Transfers returned: {len(transfers)}")
        if transfers:
            print(f"First transfer: {transfers[0]}")
    else:
        print(f"Full response: {data}")

asyncio.run(debug())