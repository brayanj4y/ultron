"""
ULTRON Face Recognition Module
Face embedding generation and comparison using InsightFace (ArcFace).
"""

import numpy as np
import cv2

# InsightFace for face embeddings
from insightface.app import FaceAnalysis


class FaceRecognizer:
    """Face recognition using ArcFace embeddings via InsightFace."""
    
    def __init__(self):
        """Initialize the face recognition model."""
        # Initialize InsightFace with buffalo_l model (good accuracy, reasonable speed)
        # Use CPU providers only for offline operation
        self.app = FaceAnalysis(
            name="buffalo_sc",  # Smaller model, faster on CPU
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=-1, det_size=(640, 480))
        self._initialized = True
    
    def get_embedding(self, face_image: np.ndarray) -> np.ndarray:
        """
        Generate face embedding from a face image.
        
        Args:
            face_image: BGR face image as numpy array
            
        Returns:
            512-dimensional face embedding, or None if no face detected
        """
        if face_image is None or face_image.size == 0:
            return None
        
        # InsightFace expects BGR images
        faces = self.app.get(face_image)
        
        if not faces:
            return None
        
        # Return embedding of the largest/most prominent face
        largest_face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])
        return largest_face.embedding
    
    def get_embedding_from_frame(self, frame: np.ndarray) -> tuple:
        """
        Get face embedding and bounding box from a full frame.
        
        Args:
            frame: BGR image as numpy array   
        Returns:
            Tuple of (embedding, bbox) or (None, None) if no face
        """
        if frame is None or frame.size == 0:
            return None, None
        
        faces = self.app.get(frame)
        
        if not faces:
            return None, None
        
        # Get the largest face
        largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        
        # Convert bbox to (x, y, w, h) format
        x1, y1, x2, y2 = largest_face.bbox.astype(int)
        bbox = (x1, y1, x2 - x1, y2 - y1)
        
        return largest_face.embedding, bbox


def compare_embeddings(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Compare two face embeddings using cosine similarity.
    
    Args:
        embedding1: First face embedding
        embedding2: Second face embedding
        
    Returns:
        Cosine similarity score (0.0 to 1.0, higher = more similar)
    """
    if embedding1 is None or embedding2 is None:
        return 0.0
    
    # Normalize embeddings
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    # Cosine similarity
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
    
    # Clamp to [0, 1] range
    return max(0.0, min(1.0, similarity))


def average_embeddings(embeddings: list) -> np.ndarray:
    """
    Average multiple face embeddings into a single representative embedding.
    
    Args:
        embeddings: List of face embeddings (numpy arrays)
        
    Returns:
        Averaged embedding, or None if empty list
    """
    if not embeddings:
        return None
    
    # Filter out None values
    valid_embeddings = [e for e in embeddings if e is not None]
    
    if not valid_embeddings:
        return None
    
    # Stack and average
    stacked = np.vstack(valid_embeddings)
    averaged = np.mean(stacked, axis=0)
    
    # Normalize the result
    norm = np.linalg.norm(averaged)
    if norm > 0:
        averaged = averaged / norm
    
    return averaged


def find_best_match(embedding: np.ndarray, authorized_users: list, threshold: float) -> tuple:
    """
    Find the best matching authorized user for a given embedding.
    
    Args:
        embedding: Face embedding to match
        authorized_users: List of (user_id, embedding, created_at) tuples
        threshold: Minimum similarity threshold
        
    Returns:
        Tuple of (user_id, similarity) if match found, else (None, 0.0)
    """
    if embedding is None or not authorized_users:
        return None, 0.0
    
    best_match = None
    best_similarity = 0.0
    
    for user_id, user_embedding, _ in authorized_users:
        similarity = compare_embeddings(embedding, user_embedding)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_match = user_id
    
    if best_similarity >= threshold:
        return best_match, best_similarity
    
    return None, best_similarity
