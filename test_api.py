import requests
import time
from PIL import Image
import io

API_URL = "http://localhost:8000"

def wait_for_server():
    print("Waiting for server to be ready...")
    for _ in range(30):  # Wait up to 300s (model load can be slow)
        try:
            response = requests.get(f"{API_URL}/health")
            if response.status_code == 200 and response.json().get("model_loaded"):
                print("Server is ready!")
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(10)
    return False

def test_generation():
    print("Testing generation endpoint...")
    # Create a dummy image (white square)
    img = Image.new('RGB', (224, 224), color='white')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    files = {'file': ('test.jpg', img_byte_arr, 'image/jpeg')}
    
    try:
        start = time.time()
        response = requests.post(f"{API_URL}/generate", files=files)
        end = time.time()
        
        if response.status_code == 200:
            tikz = response.json().get("tikz")
            print(f"Success! Generated TikZ code in {end - start:.2f}s:")
            print("-" * 20)
            print(tikz[:200] + "..." if len(tikz) > 200 else tikz)
            print("-" * 20)
        else:
            print(f"Failed with status {response.status_code}: {response.text}")
    except Exception as e:
        print(f"Error during request: {e}")

if __name__ == "__main__":
    if wait_for_server():
        test_generation()
    else:
        print("Server failed to allow connections or load model in time.")
