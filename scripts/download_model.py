"""Download explícito do detector oficial; execução normal não acessa a rede."""
import hashlib
from pathlib import Path
import urllib.request

URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
SHA256 = "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"


def main():
    target = Path(__file__).resolve().parent.parent / "assets" / "hand_landmarker.task"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == SHA256:
        print("Detector oficial já instalado e íntegro.")
        return
    with urllib.request.urlopen(URL, timeout=60) as response:
        data = response.read(32 * 1024 * 1024)
    if hashlib.sha256(data).hexdigest() != SHA256:
        raise RuntimeError("Hash do detector não confere. Download descartado; não substituí o arquivo existente.")
    temporary = target.with_suffix(".download")
    temporary.write_bytes(data)
    temporary.replace(target)
    print("Hand Landmarker instalado e SHA-256 verificado.")


if __name__ == "__main__":
    main()
