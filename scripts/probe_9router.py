import os
import sys
import json
import base64
import requests
from config import settings

print(f"Testing 9router at {settings.ROUTER_BASE_URL}")
print(f"Default model: {settings.LLM_MODEL}")

# Create a tiny 1x1 test image in PNG format
# 1x1 black pixel PNG:
TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="

def probe_text():
    url = f"{settings.ROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.ROUTER_API_KEY}"
    }
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {"role": "user", "content": "Respond with single word: OK"}
        ],
        "temperature": 0.0,
        "max_tokens": 10
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        print(f"Text probe status: {r.status_code}")
        print(f"Text probe response: {r.text[:200]}")
        return r.status_code == 200
    except Exception as e:
        print(f"Text probe error: {e}")
        return False

def probe_image():
    url = f"{settings.ROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.ROUTER_API_KEY}"
    }
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this 1x1 image? Answer in one word."},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{TINY_PNG_B64}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 20
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=25)
        print(f"Image probe status: {r.status_code}")
        print(f"Image probe response: {r.text[:300]}")
        return r.status_code == 200
    except Exception as e:
        print(f"Image probe error: {e}")
        return False

def probe_video():
    # Direct video probe: check if 9router accepts video payload or input_file
    url = f"{settings.ROUTER_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.ROUTER_API_KEY}"
    }
    # Video url / base64 video format in openai compat:
    payload = {
        "model": settings.LLM_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe video"},
                    {
                        "type": "video_url",
                        "video_url": {"url": "data:video/mp4;base64,AAAA"}
                    }
                ]
            }
        ],
        "temperature": 0.0,
        "max_tokens": 20
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=15)
        print(f"Video probe status: {r.status_code}")
        print(f"Video probe response: {r.text[:300]}")
        return r.status_code == 200
    except Exception as e:
        print(f"Video probe error: {e}")
        return False

if __name__ == "__main__":
    t_ok = probe_text()
    i_ok = probe_image()
    v_ok = probe_video()
    print("\nPROBE SUMMARY:")
    if v_ok:
        print("CLASSIFICATION: DIRECT_VIDEO_SUPPORTED")
    elif i_ok:
        print("CLASSIFICATION: MULTIMODAL_IMAGE_SUPPORTED")
    elif t_ok:
        print("CLASSIFICATION: TEXT_ONLY")
    else:
        print("CLASSIFICATION: ROUTER_UNAVAILABLE")
