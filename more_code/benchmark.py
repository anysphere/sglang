import aiohttp
import asyncio
import time
import argparse
import os
import random
import numpy as np
from fireworks.client import Fireworks

# 100 token DeepSeek-V2 Lite prompt
BASE_PROMPT = "You are a master chef specializing in French cuisine. Help aspiring cooks learn classic recipes. Break down complex methods into simple steps. Share insider tips from professional kitchens. Focus on teaching fundamental skills that build confidence. Be encouraging but maintain high standards for proper technique. Ensure the food is delicious and vegetarian. Use plenty of healthy ingredients that are nutritious and sustainable but are still flavorful. Be very verbose! I will repeat this instruction many times so that you understand it! "

i = 0

async def measure_single_request(
    client: Fireworks | None,
    api: str,
    model: str,
    latencies: list[float],
    input_tokens: int,
    output_tokens: int
) -> float:
    # Generate fixed size prompt
    if input_tokens % 100 != 0:
        raise ValueError(f"Input tokens must be a multiple of 100 but got {input_tokens}")
    # Add a random prefix to the prompt to avoid caching
    prompt = "".join([''.join(random.choices('0123456789', k=8)) + BASE_PROMPT] * (input_tokens // 100))
    
    start_time = time.time()
    try:
        global i
        i += 1
        if api == "fireworks":
            # TODO: verify that it generates enough tokens
            async for _ in client.completions.acreate(
                model=model,
                prompt=prompt,
                max_tokens=output_tokens,
            ):
                pass
        elif api == "sglang":
            async with aiohttp.ClientSession() as session:
                await session.post(
                    "http://localhost:30000/generate",
                    json={
                        "text": prompt,
                        "sampling_params": {
                            "max_new_tokens": output_tokens,
                            "ignore_eos": True,
                        },
                    },
                )
        else:
            raise ValueError(f"Unknown API provider: {api}")
        latencies.append(time.time() - start_time)
    except Exception as e:
        print(f"Request failed: {e}")
        return None

async def test_at_rps(
    client: Fireworks | None,
    api: str,
    model: str,
    target_rps: float,
    input_tokens: int,
    output_tokens: int,
    duration: int,
    warmup: int,
    cooldown: int,
    latencies: list[float]
) -> None:
    interval = 1.0 / target_rps
    total_duration = warmup + duration + cooldown
    start_time = time.time()
    next_request_time = start_time
    futures = []

    while time.time() - start_time < total_duration:
        current_time = time.time()
        futures.append(asyncio.ensure_future(measure_single_request(client, api, model, latencies, input_tokens, output_tokens)))
        next_request_time = current_time + interval

        if time.time() > next_request_time:
            raise RuntimeError(f"Request queueing is too slow. {time.time()=} {next_request_time=}")

        # Wait for next interval
        await asyncio.sleep(next_request_time - time.time())

    await asyncio.gather(*futures)

def main():
    parser = argparse.ArgumentParser(description='Generate latency-throughput curve for LLM endpoint')
    parser.add_argument('--api-key', type=str, default=os.getenv("FIREWORKS_API_KEY"), help='API key')
    parser.add_argument('--api', type=str, default='fireworks', help='API provider')
    parser.add_argument('--rps', type=float, nargs='+', default=[1, 2, 4, 8, 16], help='List of RPS values to test')
    parser.add_argument('-i', '--input-tokens', type=int, default=100, help='Number of input tokens per request')
    parser.add_argument('-o', '--output-tokens', type=int, default=100, help='Number of output tokens per request')
    parser.add_argument('--duration', type=int, default=60, help='Measurement duration in seconds')
    parser.add_argument('--warmup', type=int, default=30, help='Warmup duration in seconds')
    parser.add_argument('--cooldown', type=int, default=10, help='Cooldown duration in seconds')
    parser.add_argument('--model', type=str, default="anysphere/cpp-dsv2-10-19-maybe-ctx-fixed#anysphere/d4d11091", help='Model identifier')
    
    args = parser.parse_args()
    client = None
    if args.api == "fireworks":
        client = Fireworks(api_key=args.api_key)
    elif args.api == "sglang":
        client = None
    else:
        raise ValueError(f"Unknown API provider: {args.api}")
    
    latencies = []
    throughputs = []
    tpots = []
    concurrencies = []
    
    for rps in args.rps:
        print(f"\nTesting RPS: {rps}")
        rps_latencies = []
        
        asyncio.run(test_at_rps(
            client=client,
            api=args.api,
            model=args.model,
            target_rps=rps,
            input_tokens=args.input_tokens,
            output_tokens=args.output_tokens,
            duration=args.duration,
            warmup=args.warmup,
            cooldown=args.cooldown,
            latencies=rps_latencies,
        ))
        
        # TODO: Validate mean calculation -- is median a good enough approximation? is value stable?
        # Slice off warmup and cooldown periods where concurrency for target RPS is not achieved
        # print(f'{latencies=}')
        valid_latencies = rps_latencies[int(args.warmup * rps):-int(args.cooldown * rps)]
        if len(valid_latencies) > 0:
            # TODO: Doesn't compute diverging where latencies increase as server is backlogged
            median_latency = np.median(np.array(valid_latencies))
            throughput = rps * (args.input_tokens + args.output_tokens)
            tpot = rps * args.output_tokens
            print(f"Result - Latency: {median_latency:.3f}s, Throughput: {throughput:.1f} tokens/s, Tpot: {tpot:.1f} tokens/s")
            latencies.append(median_latency)
            throughputs.append(throughput)
            tpots.append(tpot)
            concurrencies.append(median_latency * rps)
        else:
            print("No valid measurements collected")

    print('Latency\tThroughput\tTpot\tConcurrency')
    for latency, throughput, tpot, concurrency in zip(latencies, throughputs, tpots, concurrencies):
        print(f"{latency:.3f}\t{throughput:.1f}\t\t{tpot:.1f}\t{concurrency:.1f}")

if __name__ == "__main__":
    main()