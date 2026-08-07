# ultron offline security

offline laptop security system. face recognition. surveillance. voice override.
hardware integration. arduino. keypad. servo.

---

## 0. features
offline. no internet. local processing.
registration. captures 20 face samples.
duty mode. continuous surveillance.
threat detection. logs unrecognized people.
voice override. speak code to authorize.
hardware lock. keypad entry. servo control.

## 1. requirements
python 3.8+
webcam / mic / speakers
2gb disk space

**hardware**
arduino uno
servo motor (pin 10)
4x4 keypad (pins 2-9)

## 2. setup

### software
```bash
pip install -r requirements.txt
```
download `vosk-model-small-en-us-0.15` -> rename to `vosk-model`.

### hardware
upload `ultron arduino/ultron_arduino.ino` to arduino.
connect:
- servo signal -> pin 10
- keypad rows -> pins 2,3,4,5
- keypad cols -> pins 6,7,8,9

## 3. usage

start system:
```bash
python main.py
```

### [modes]
1 registration -> capture face samples.
2 duty mode -> start surveillance.
3 view users -> list database.

### [access]
face unlock -> camera sees authorized user -> door unlocks.
keypad unlock -> enter `1234` or `#` -> door unlocks.
voice override -> threat detected? say `ultron override` -> door unlocks.

## 4. configuration
edit `config.yaml`.

```yaml
face_match_threshold: 0.6
registration_samples: 20
admin_passphrase: "ultron override"
voice_volume: 1.0
camera_index: 0
admin_code_timeout: 10
arduino_port: "COM3"
```

## 5. status
vosk model not found -> check folder name `vosk-model`.
arduino missing -> check usb connection + config `COM` port.
no audio -> ensures `tts_worker.py` exists (subprocess mode).

## 6. security
data is local.
sqlite database.
raw images stored.
no network.
keep `config.yaml` safe.

---
license: mit. use at own risk.
