import os, json
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

print("=== OpenAI Batches ===")
try:
    openai_client = OpenAI()
    for batch in openai_client.batches.list(limit=10):
        print(f"ID: {batch.id} | Status: {batch.status} | Created: {batch.created_at}")
        if batch.status == 'failed' or batch.status == 'expired':
            if batch.errors:
                print(f"  Errors: {batch.errors}")
except Exception as e:
    print(f"OpenAI error: {e}")

print("\n=== Anthropic Batches ===")
try:
    anthropic_client = Anthropic()
    for batch in anthropic_client.beta.messages.batches.list(limit=10) if hasattr(anthropic_client, 'beta') else anthropic_client.messages.batches.list(limit=10):
        print(f"ID: {batch.id} | Status: {batch.processing_status} | Created: {batch.created_at}")
except Exception as e:
    print(f"Anthropic error: {e}")
