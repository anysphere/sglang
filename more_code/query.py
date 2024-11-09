import requests
import asyncio
import argparse

async def query_server(host: str, port: int, prompt: str, predicted: str = None):
    url = f"http://{host}:{port}/generate"
    
    # Prepare request payload
    payload = {
        "text": "The capital of France is",
        "sampling_params": {
            "temperature": 0.9,
            "max_new_tokens": 32,
        },
    }

    if predicted is not None:
        payload["predicted_output"] = predicted

    response = requests.post(
        f"http://{host}:{port}/generate",
        json=payload,
    )

    print(response.json())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--predicted", type=str, help="Optional predicted text for speculation")
    args = parser.parse_args()

    asyncio.run(query_server(args.host, args.port, args.prompt, args.predicted))

if __name__ == "__main__":
    main()