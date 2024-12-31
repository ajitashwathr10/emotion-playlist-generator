import os
import json
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import spacy
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    Pipeline
)
from torch.nn import functional as F
import logging
from enum import Enum
import boto3
from botocore.exceptions import ClientError

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger(__name__)

class StorytellingException(Exception):
    pass

class ProcessingError(StorytellingException):
    pass

class LayoutError(StorytellingException):
    pass

class ImageGenerationError(StorytellingException):
    pass

class StorageError(StorytellingException):
    pass

@dataclass
class Entity:
    name: str
    type: str
    attributes: Dict[str, str]
    position: Optional[Dict[str, float]] = None
    importanc: float = 1.0

    def to_dict(self) -> Dict:
        return asdict(self)
    
class EmotionType(Enum):
    HAPPY = "happy"
    SAD = "sad"
    NEUTRAL = "neutral"
    ANGRY = "angry"
    SURPRISE = "surprise"
    FEAR = "fear"
    DISGUST = "disgust"
    CONFUSED = "confused"
    OTHER = "other"

    @classmethod
    def from_string(cls, emotion: str) -> 'EmotionType':
        try:
            return cls(emotion.lower())
        except ValueError:
            return cls.OTHER
@dataclass
class Scene:
    id: int
    text: str
    entities: List[Entity]
    mood: EmotionType
    actions: List[str]
    layout: Optional[Dict] = None
    image_path: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'text': self.text,
            'entities': [e.to_dict() for e in self.entities],
            'mood': self.mood.value,
            'actions': self.actions,
            'layout': self.layout,
            'image_path': self.image_path
        }

class TextProcesor:
    def __init__(self):
        try:
            self.nlp = spacy.load("en_core_web_lg")
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.emotion_model = AutoModelForSequenceClassification.from_pretrained("microsoft/bert-base-uncased-emotion")
            logging.info("Successfully initialized NLP models")
        except Exception as e:
            logging.error(f"Failed to initialize NLP models: {e}")
            raise ProcessingError("Failed to initialize NLP models")
    def segment_scenes(self, text: str) -> List[str]:
        try: 
            doc = self.nlp(text)
            scenes = []
            current_scene = []
            for sent in doc.sents:
                current_scene.append(sent.text)
                if any(marker in sent.text.lower() for marker in
                       ["later", "meanwhile", "after", "before", "next", "suddenly"]):
                    scenes.append(" ".join(current_scene))
                    current_scene = []
            if current_scene:
                scenes.append(" ".join(current_scene))
            return scenes if scenes else [text]
        except Exception as e:
            logging.error(f"Failed to segment scenes: {e}")
            raise ProcessingError("Failed to segment scenes")
        
    def extract_entities(self, text: str) -> List[Entity]:
        try:
            doc = self.nlp(text)
            entities = []
            for ent in doc.ents:
                importance = 1.0
                if ent.label_ in ["PERSON", "ORG"]:
                    importance *= 1.5
                attributes = {
                    "description": self._get_entity_description(ent, doc),
                    "label": ent.label_
                }
                entities.append(Entity(
                    name = ent.text,
                    type = ent.label_,
                    attributes = attributes,
                    importance = importance
                ))
            return entities
        except Exception as e:
            logging.error(f"Entity extraction failed: {str(e)}")
            raise ProcessingError(f"Entity extraction failed: {str(e)}")
    
    def _get_entity_description(self, ent, doc) -> str:
        description_terms = []
        for token in doc:
            if token.head == ent.root:
                if token.dep_ in ["amod", "advmod"]:
                    description_terms.append(token.text)
        return " ".join(description_terms) if description_terms else ""
    
    def detect_actions(self, text: str) -> List[str]:
        try:
            doc = self.nlp(text)
            actions = []
            for token in doc:
                if token.pos_ == "VERB" and token.dep_ in ["ROOT", "xcomp"]:
                    verb_phrase = self._extract_verb_phrase(token)
                    if verb_phrase:
                        actions.append(verb_phrase)
            return actions
        except Exception as e:
            logger.error(f"Action detection failed: {str(e)}")
            raise ProcessingError(f"Action detection failed: {str(e)}")

    def _extract_verb_phrase(self, verb_token) -> str:
        phrase_tokens = [verb_token.text]
        for child in verb_token.children:
            if child.dep_ in ["dobj", "pobj"]:
                phrase_tokens.append(child.text)
        return " ".join(phrase_tokens)

    def analyze_mood(self, text: str) -> EmotionType:
        try:
            inputs = self.tokenizer(text, return_tensors = "pt", truncation = True, max_length = 512)
            outputs = self.emotion_model(**inputs)
            emotion_probs = F.softmax(outputs.logits, dim = 1)

            emotion_mapping = {
                0: EmotionType.ANGRY,
                1: EmotionType.HAPPY,
                2: EmotionType.SAD,
                3: EmotionType.DISGUST,
                4: EmotionType.FEAR,
                5: EmotionType.NEUTRAL,
                6: EmotionType.SURPRISE,
                7: EmotionType.CONFUSED
            }
            predicted_emotion = emotion_mapping[torch.argmax(emotion_probs).item()]
            return predicted_emotion
        except Exception as e:
            logger.error(f"Mood analysis failed: {str(e)}")
            raise ProcessingError(f"Mood analysis failed: {str(e)}")

class LayoutEngine:
    def __init__(self):
        self.composition_rules = self.load_composition_rules()
        self.grid_size = (8, 8)

    def load_composition_rules(self) -> Dict:
        try:
            rules = {
                "rule_of_thirds": True,
                "golden_ratio": True,
                "min_spacing": 0.1,
                "focus_points": [
                    (0.333, 0.333),
                    (0.333, 0.667),
                    (0.667, 0.333),
                    (0.667, 0.667)
                ],
                "margins": {
                    "top": 0.1,
                    "bottom": 0.1,
                    "left": 0.1,
                    "right": 0.1
                }
            }
            return rules
        except Exception as e:
            logger.error(f"Failed to load composition rules: {str(e)}")
            raise LayoutError("Failed to load composition rules")

    def generate_layout(self, scene: Scene) -> Dict:
        try:
            layout = {
                "frame_size": (1024, 1024),
                "entities": {},
                "focal_point": None,
                "depth_layers": []
            }
            sorted_entities = sorted(
                scene.entities,
                key = lambda x: x.importance,
                reverse = True
            )
            depth_layers = self._create_depth_layers(sorted_entities)
            layout["depth_layers"] = depth_layers
            occupied_positions = set()
            for entity in sorted_entities:
                position = self._calculate_optimal_position(
                    entity,
                    occupied_positions,
                    layout['frame_size']
                )
                layout["entites"][entity.name] = {
                    "position": position,
                    "scale": self._calculate_scale(entity, depth_layers),
                    "rotation": self._calculate_rotation(entity, scene.actions)
                }
                occupied_positions.add(position["x"], position["y"])
            if scene.actions:
                layout["focal_point"] = self._determine_focal_point(
                    scene.actions[0],
                    layout["entities"]
                )
            return layout
        except Exception as e:
            logger.error(f"Layout generation failed: {str(e)}")
            raise LayoutError(f"Layout generation failed: {str(e)}")
    
    def _create_depth_layers(self, entities: List[Entity]) -> List[List[str]]:
        layers = [[], [], []]
        for entity in entities:
            if entity.type in ["PERSON", "ORG"] and entity.importance > 1.2:
                layers[0].append(entity.name)
            elif entity.type in ["PERSON", "ORG", "PRODUCT"]:
                layers[1].append(entity.name)
            else:
                layers[2].append(entity.name)
        return layers
    
    def _calculate_optimal_position(
        self, 
        entity: Entity,
        occupied_position: set,
        frame_size: Tuple[int, int]
    ) -> Dict[str, float]:
        grid = np.zero(self.grid_size)
        for pos in occupied_position:
            grid_x = int(pos[0] * self.grid[0])
            grid_y = int(pos[1] * self.grid[1])
            grid[grid_y, grid_x] = 1
        scores = np.zeros_like(grid)
        for y in range(self.grid_size[0]):
            for x in range(self.grid_size[1]):
                if grid[y, x] == 0:
                    scores[y, x] = self._calculate_optimal_position(
                        x / self.grid_size[0],
                        y / self.grid_size[1],
                        entity
                    )
        best_pos = np.unravel_index(np.argmax(scores), scores.shape)
        return {
            "x": best_pos[1] / self.grid_size[0],
            "y": best_pos[0] / self.grid_size[1]
        }

    def _calculate_position_score(
        self,
        x: float,
        y: float,
        entity: Entity
    ) -> float:
        score = 0.0
        if self.composition_rules["rule_of_thirds"]:
            for third in [0.333, 0.667]:
                score += (1 - min(abs(x - third), abs(y - third))) * 0.5
        if self.composition_rules["golden_ratio"]:
            golden_ratio = 0.618
            golden_points = [
                (golden_ratio, golden_ratio),
                (1 - golden_ratio, golden_ratio),
                (golden_ratio, 1- golden_ratio),
                (1 - golden_ratio, 1 - golden_ratio)
            ]
            for px, py in golden_points:
                distance = np.sqrt((x - px) ** 2 + (y - py) ** 2)
                score += (1 - distance) * 0.3
        score *= entity.importance
        return score
    
    def _calculate_scale(
        self, 
        entity: Entity,
        depth_layers: List[List[str]]
    ) -> float:
        if entity.name in depth_layers[0]:
            return 1.0 * entity.importance
        elif entity.name in depth_layers[1]:
            return 0.7 * entity.importance
        else:
            return 0.4 * entity.importance
    
    def _calculate_rotation(
        self,
        entity: Entity,
        actions: List[str]
    ) -> float:
        rotation = 0.0
        for action in actions:
            if entity.name.lower() in action.lower():
                if any(term in action.lower() for term in ["turn", "spin", "rotate"]):
                    rotation = 45.0
        return rotation
    
    def _determine_focal_point(
        self,
        main_action: str,
        entity_layouts: Dict
    ) -> Dict[str, float]:
        involved_entities = [
            name for name in entity_layouts.keys()
            if name.lower() in main_action.lower()
        ]
        if involved_entities:
            x_sum = sum(entity_layouts[e]["position"]["x"] for e in involved_entities)
            y_sum = sum(entity_layouts[e]["position"]["y"] for e in involved_entities)
            return {
                "x": x_sum / len(involved_entities),
                "y": y_sum / len(involved_entities)
            }
        return {"x": 0.5, "y": 0.5}

class ImageGenerator:
    def __init__(self, model_path: Optional[str] = None):
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = self._load_model(model_path)
            logger.info("Successfully initialized image generation model")
        except Exception as e:
            logger.error(f"Failed to initialize image generation model: {str(e)}")
            raise ImageGenerationError("Failed to initialize image generation model")
    
    def _load_model(self, model_path: Optional[str]) -> Any:
        


    


