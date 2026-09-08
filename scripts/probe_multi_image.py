import base64
import requests
import cv2
import numpy as np
from config import settings

# Create 2 small 160x90 test frames
frame1 = np.zeros((90, 160, 3), dtype=np.uint8)
cv2.putText(frame1, "Frame 1", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
_, buf1 = cv2.imencode(".jpg", frame1)
b64_1 = base64.b64encode(buf1).decode("utf-8")

frame2 = np.zeros((90, 160, 3), dtype=np.uint8)
cv2.putText(frame2, "Frame 2", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
_, buf2 = cv2.imencode(".jpg", frame2)
b64_2 = base64.b64encode(buf2).decode("utf-8")

# Test 1: Multiple image_url items in one user message
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
                {"type": "text", "text": "Analyze these two frames. What text is written in Frame 1 and Frame 2? Reply in JSON."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_1}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_2}"}}
            ]
        }
    ],
    "temperature": 0.0,
    "stream": False
}

try:
    r = requests.post(url, headers=headers, json=payload, timeout=30)
    print("Multi-image status:", r.status_code)
    if r.status_code == 200:
        j = r.json()
        print("Response:", j["choices"][0]["message"]["content"])
    else:
        print("Error response:", r.text[:300])
except Exception as e:
    print("Multi-image exception:", e)
