"""
ULTRON - Offline Security System
Main entry point with Registration and Duty modes.
"""

import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import yaml
import numpy as np

# ULTRON modules
import database
from camera import Camera
from face_recognition import FaceRecognizer, average_embeddings, find_best_match
from tts import create_tts, MESSAGES
from stt import create_stt
from arduino_interface import create_arduino


# Paths
BASE_DIR = Path(__file__).parent
CONFIG_FILE = BASE_DIR / "config.yaml"
AUTHORIZED_DIR = BASE_DIR / "authorized_faces"
UNAUTHORIZED_DIR = BASE_DIR / "unauthorized_faces"
LOGS_DIR = BASE_DIR / "logs"


def load_config() -> dict:
    """Load configuration from YAML file."""
    if not CONFIG_FILE.exists():
        print(f"ERROR: Config file not found at {CONFIG_FILE}")
        sys.exit(1)
    
    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def log_event(message: str):
    """Log an event with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    
    # Also write to log file
    log_file = LOGS_DIR / f"ultron_{datetime.now().strftime('%Y%m%d')}.log"
    with open(log_file, "a") as f:
        f.write(log_line + "\n")


def save_face_image(frame: np.ndarray, directory: Path, prefix: str = "face") -> Path:
    """Save a face image to the specified directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.jpg"
    filepath = directory / filename
    cv2.imwrite(str(filepath), frame)
    return filepath


class UltronSystem:
    """Main ULTRON security system."""
    
    def __init__(self, config: dict):
        """Initialize ULTRON with configuration."""
        self.config = config
        
        # Initialize components
        log_event("Initializing ULTRON components...")
        
        self.camera = Camera(camera_index=config.get("camera_index", 0))
        self.face_recognizer = FaceRecognizer()
        self.tts = create_tts(
            volume=config.get("voice_volume", 1.0),
            rate=config.get("voice_rate", 150)
        )
        
        # STT is loaded on demand (needs Vosk model)
        self.stt = None
        
        # Initialize database
        database.init_db()
        
        # Initialize Arduino
        try:
            self.arduino = create_arduino(config.get("arduino_port", "COM3"))
            log_event("Arduino interface initialized.")
        except Exception as e:
            log_event(f"WARNING: Arduino init failed: {e}")
            self.arduino = None
        
        # Configuration values
        self.threshold = config.get("face_match_threshold", 0.6)
        self.registration_samples = config.get("registration_samples", 20)
        self.admin_passphrase = config.get("admin_passphrase", "ultron override")
        self.admin_code_timeout = config.get("admin_code_timeout", 10)
        
        log_event("ULTRON initialized successfully.")
    
    def _init_stt(self):
        """Initialize STT on demand."""
        if self.stt is None:
            try:
                self.stt = create_stt()
                log_event("Speech recognition initialized.")
            except FileNotFoundError as e:
                log_event(f"WARNING: {e}")
                self.stt = None
    
    def run_registration(self) -> bool:
        """
        Run registration mode to capture and store a user's face.
        
        Returns:
            True if registration successful
        """
        log_event("=== REGISTRATION MODE ===")
        self.tts.speak(MESSAGES["registration_start"])
        
        if not self.camera.start():
            log_event("ERROR: Failed to start camera")
            return False
        
        # Get user ID
        user_id = input("Enter user ID (or press Enter for auto-generated): ").strip()
        if not user_id:
            user_id = f"user_{uuid.uuid4().hex[:8]}"
        
        if database.user_exists(user_id):
            log_event(f"User '{user_id}' already exists!")
            self.camera.release()
            return False
        
        print(f"\nRegistering user: {user_id}")
        print(f"Will capture {self.registration_samples} face samples.")
        print("Position your face in the camera. Press 'q' to cancel.\n")
        
        embeddings = []
        captured = 0
        last_capture_time = 0
        capture_interval = 0.5  # seconds between captures
        
        try:
            while captured < self.registration_samples:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                # Get face embedding
                embedding, bbox = self.face_recognizer.get_embedding_from_frame(frame)
                
                # Display frame
                display_frame = frame.copy()
                if bbox:
                    x, y, w, h = bbox
                    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Captured: {captured}/{self.registration_samples}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow("ULTRON Registration", display_frame)
                
                # Capture embedding if face detected and enough time passed
                current_time = time.time()
                if embedding is not None and (current_time - last_capture_time) >= capture_interval:
                    embeddings.append(embedding)
                    captured += 1
                    last_capture_time = current_time
                    log_event(f"Captured sample {captured}/{self.registration_samples}")
                    
                    # Save face image
                    if bbox:
                        x, y, w, h = bbox
                        face_img = frame[max(0,y):y+h, max(0,x):x+w]
                        save_face_image(face_img, AUTHORIZED_DIR, f"{user_id}_sample{captured}")
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log_event("Registration cancelled by user.")
                    cv2.destroyAllWindows()
                    self.camera.release()
                    return False
        
        finally:
            cv2.destroyAllWindows()
            self.camera.release()
        
        if len(embeddings) < 5:  # Minimum samples needed
            log_event("Not enough face samples captured. Registration failed.")
            return False
        
        # Average embeddings and store
        avg_embedding = average_embeddings(embeddings)
        if avg_embedding is None:
            log_event("Failed to generate embedding. Registration failed.")
            return False
        
        # Store in database
        if database.add_user(user_id, avg_embedding):
            log_event(f"User '{user_id}' registered successfully with {len(embeddings)} samples.")
            self.tts.speak(MESSAGES["registration_complete"])
            return True
        else:
            log_event("Failed to store user in database.")
            return False
    
    def run_duty_mode(self):
        """
        Run duty (surveillance) mode.
        Continuously monitor for faces and identify threats.
        """
        log_event("=== DUTY MODE ===")
        
        # Check if there are registered users
        user_count = database.count_users()
        if user_count == 0:
            log_event("No authorized faces registered. Cannot start duty mode.")
            self.tts.speak(MESSAGES["no_faces_registered"])
            return
        
        log_event(f"Loaded {user_count} authorized user(s).")
        self.tts.speak(MESSAGES["duty_start"])
        
        if not self.camera.start():
            log_event("ERROR: Failed to start camera")
            return
        
        # Load authorized users
        authorized_users = database.get_all_users()
        
        print("\nDuty mode active. Press 'q' to exit.\n")
        
        last_threat_time = 0
        threat_cooldown = 5  # seconds between threat alerts
        
        try:
            while True:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                # Get face embedding
                embedding, bbox = self.face_recognizer.get_embedding_from_frame(frame)
                
                # Display frame
                display_frame = frame.copy()
                status_text = "Scanning..."
                status_color = (255, 255, 255)
                
                if embedding is not None and bbox is not None:
                    x, y, w, h = bbox
                    
                    # Check against authorized users
                    matched_user, similarity = find_best_match(
                        embedding, authorized_users, self.threshold
                    )
                    
                    if matched_user:
                        # Authorized
                        status_text = f"AUTHORIZED: {matched_user} ({similarity:.2f})"
                        status_color = (0, 255, 0)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    else:
                        # Threat detected
                        current_time = time.time()
                        status_text = f"THREAT DETECTED ({similarity:.2f})"
                        status_color = (0, 0, 255)
                        cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
                        
                        # Handle threat if cooldown passed
                        if current_time - last_threat_time > threat_cooldown:
                            last_threat_time = current_time
                            self._handle_threat(frame, bbox, embedding)
                            # Reload users in case new one was added
                            authorized_users = database.get_all_users()
                
                # Show status
                cv2.putText(display_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                cv2.putText(display_frame, "Press 'q' to exit", (10, display_frame.shape[0] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                
                cv2.imshow("ULTRON Duty Mode", display_frame)
                
                # Check for quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    log_event("Duty mode ended by user.")
                    break

                # Check Arduino Keypad
                # Only check if threat detected or just periodically to allow manual unlock?
                # Let's allow manual unlock via code at any time
                if self.arduino:
                    keypad_input = self.arduino.check_keypad()
                    if keypad_input:
                        log_event(f"Keypad input: {keypad_input}")
                        # Simple logic: If any key pressed, treat as part of code
                        # Ideally, buffer keys until '#' is pressed
                        # For now, let's keep it simple or implement a quick buffer check
                        self._handle_keypad_input(keypad_input)
        
        finally:
            cv2.destroyAllWindows()
            self.camera.release()
            if self.arduino:
                self.arduino.lock_door() # Ensure locked on exit

    def _handle_keypad_input(self, key):
        """Handle input from Arduino keypad."""
        # Simple buffer stored in instance
        if not hasattr(self, '_keypad_buffer'):
            self._keypad_buffer = ""
        
        if key == '#': # Enter
            code = self._keypad_buffer
            log_event(f"Checking keypad code: {code}")
            if code == "1234": # Hardcoded for now, or use admin_passphrase logic
                log_event("Keypad Access Granted")
                self.tts.speak(MESSAGES["authorized"])
                self.arduino.unlock_door()
                time.sleep(3) # Keep open for 3 seconds
                self.arduino.lock_door()
            else:
                log_event("Keypad Access Denied")
                self.tts.speak(MESSAGES["admin_rejected"])
            self._keypad_buffer = ""
        elif key == '*': # Clear
            self._keypad_buffer = ""
            log_event("Keypad buffer cleared")
        else:
            self._keypad_buffer += key
            # Beep or feedback could go here
    
    def _handle_threat(self, frame: np.ndarray, bbox: tuple, embedding: np.ndarray):
        """Handle a detected threat."""
        log_event("THREAT DETECTED - Unrecognized individual")
        
        # Save unauthorized face
        x, y, w, h = bbox
        face_img = frame[max(0,y):y+h, max(0,x):x+w]
        save_path = save_face_image(face_img, UNAUTHORIZED_DIR, "threat")
        log_event(f"Threat face saved to: {save_path}")
        
        # Speak warning
        self.tts.speak(MESSAGES["threat_detected"])
        
        # Listen for admin code
        self._init_stt()
        if self.stt is None:
            log_event("Speech recognition not available. Skipping admin code check.")
            return
        
        log_event(f"Listening for admin code ({self.admin_code_timeout}s timeout)...")
        self.tts.speak(MESSAGES["listening"])
        
        # Create grammar from admin passphrase + common variants
        admin_words = self.admin_passphrase.lower().split()
        grammar = admin_words + ["admin", "override", "code", "ultron", "access"]
        
        spoken = self.stt.listen_for_phrase(timeout=self.admin_code_timeout, grammar=grammar)
        
        if spoken:
            log_event(f"Heard: '{spoken}'")
            
            if self.stt.check_admin_code(spoken, self.admin_passphrase):
                log_event("Admin code ACCEPTED")
                self.tts.speak(MESSAGES["admin_accepted"])
                
                # Add this face to authorized database
                new_user_id = f"authorized_{uuid.uuid4().hex[:8]}"
                if database.add_user(new_user_id, embedding):
                    log_event(f"New user '{new_user_id}' added to authorized database.")
                    # Save to authorized faces
                    save_face_image(face_img, AUTHORIZED_DIR, new_user_id)
            else:
                log_event("Admin code REJECTED")
                self.tts.speak(MESSAGES["admin_rejected"])
        else:
            log_event("No speech detected during admin code window.")
    
    def shutdown(self):
        """Shutdown ULTRON system."""
        log_event("Shutting down ULTRON...")
        self.tts.speak(MESSAGES["shutting_down"])
        self.camera.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    print("=" * 50)
    print("         ULTRON SECURITY SYSTEM")
    print("=" * 50)
    print()
    
    # Load configuration
    config = load_config()
    
    # Initialize system
    ultron = UltronSystem(config)
    ultron.tts.speak(MESSAGES["startup"])
    
    try:
        while True:
            print("\n" + "=" * 40)
            print("ULTRON MAIN MENU")
            print("=" * 40)
            print("1. Registration Mode (Add authorized user)")
            print("2. Duty Mode (Surveillance)")
            print("3. View Registered Users")
            print("4. Exit")
            print("-" * 40)
            
            choice = input("Select option: ").strip()
            
            if choice == "1":
                ultron.run_registration()
            
            elif choice == "2":
                ultron.run_duty_mode()
            
            elif choice == "3":
                users = database.get_all_users()
                if users:
                    print(f"\nRegistered Users ({len(users)}):")
                    for user_id, _, created_at in users:
                        print(f"  - {user_id} (registered: {created_at})")
                else:
                    print("\nNo users registered.")
            
            elif choice == "4":
                break
            
            else:
                print("Invalid option. Please try again.")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    
    finally:
        ultron.shutdown()


if __name__ == "__main__":
    main()
