"""Decision Engine for Constellation X.

Analyzes HydraNet V2 outputs and generates driving decisions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import numpy as np


class Action(Enum):
    """Driving actions."""
    MAINTAIN = "MAINTAIN"  # Keep current speed
    SLOW = "SLOW"          # Reduce speed
    STOP = "STOP"          # Full stop required
    CAUTION = "CAUTION"    # Proceed with caution


@dataclass
class Detection:
    """Single detection result."""
    class_name: str
    class_id: int
    box: Tuple[float, float, float, float]  # x1, y1, x2, y2
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class Decision:
    """Driving decision output."""
    action: Action
    reason: str
    confidence: float
    priority_detections: List[Detection]  # Objects that influenced decision
    road_percentage: float
    sidewalk_percentage: float


class DecisionEngine:
    """Analyzes perception outputs and generates driving decisions."""

    # Class names from Cityscapes
    CLASS_NAMES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

    # Priority classes (pedestrians/cyclists are highest priority)
    VULNERABLE_CLASSES = {"person", "rider", "bicycle"}
    VEHICLE_CLASSES = {"car", "truck", "bus", "train", "motorcycle"}

    def __init__(
        self,
        image_size: Tuple[int, int] = (512, 1024),  # H, W
        danger_zone_ratio: float = 0.4,  # Center 40% of image width
        close_threshold: float = 0.15,   # Object height > 15% of image = close
        min_road_ratio: float = 0.1,     # Minimum road visibility
    ):
        self.image_height, self.image_width = image_size
        self.danger_zone_ratio = danger_zone_ratio
        self.close_threshold = close_threshold
        self.min_road_ratio = min_road_ratio

        # Define danger zone (center of image)
        margin = (1 - danger_zone_ratio) / 2
        self.danger_zone_left = self.image_width * margin
        self.danger_zone_right = self.image_width * (1 - margin)

    def analyze(
        self,
        detections: List[Detection],
        segmentation_mask: np.ndarray,
    ) -> Decision:
        """Analyze perception outputs and generate decision.

        Args:
            detections: List of Detection objects
            segmentation_mask: (H, W) numpy array with class indices
                              0=background, 1=road, 2=sidewalk

        Returns:
            Decision object with action, reason, and supporting data
        """
        # Compute segmentation stats
        total_pixels = segmentation_mask.size
        road_pixels = (segmentation_mask == 1).sum()
        sidewalk_pixels = (segmentation_mask == 2).sum()
        road_percentage = road_pixels / total_pixels
        sidewalk_percentage = sidewalk_pixels / total_pixels

        # Categorize detections
        vulnerable_in_danger = []
        vehicles_in_danger = []
        close_objects = []

        for det in detections:
            # Check if in danger zone (center of image)
            cx, cy = det.center
            in_danger_zone = self.danger_zone_left < cx < self.danger_zone_right

            # Check if close (large in frame)
            is_close = det.height > (self.image_height * self.close_threshold)

            if in_danger_zone:
                if det.class_name in self.VULNERABLE_CLASSES:
                    vulnerable_in_danger.append(det)
                elif det.class_name in self.VEHICLE_CLASSES:
                    vehicles_in_danger.append(det)

            if is_close:
                close_objects.append(det)

        # Decision logic (priority order)

        # 1. STOP: Vulnerable road user close and in danger zone
        vulnerable_close = [d for d in vulnerable_in_danger if d in close_objects]
        if vulnerable_close:
            priority_det = max(vulnerable_close, key=lambda d: d.area)
            return Decision(
                action=Action.STOP,
                reason=f"{priority_det.class_name.upper()} detected directly ahead ({priority_det.confidence:.0%} conf)",
                confidence=priority_det.confidence,
                priority_detections=vulnerable_close,
                road_percentage=road_percentage,
                sidewalk_percentage=sidewalk_percentage,
            )

        # 2. SLOW: Vehicle close and in danger zone
        vehicles_close = [d for d in vehicles_in_danger if d in close_objects]
        if vehicles_close:
            priority_det = max(vehicles_close, key=lambda d: d.area)
            return Decision(
                action=Action.SLOW,
                reason=f"{priority_det.class_name.upper()} ahead - maintain distance",
                confidence=priority_det.confidence,
                priority_detections=vehicles_close,
                road_percentage=road_percentage,
                sidewalk_percentage=sidewalk_percentage,
            )

        # 3. CAUTION: Vulnerable road users visible (not immediately dangerous)
        if vulnerable_in_danger:
            priority_det = max(vulnerable_in_danger, key=lambda d: d.confidence)
            return Decision(
                action=Action.CAUTION,
                reason=f"{priority_det.class_name} visible in path - proceed carefully",
                confidence=priority_det.confidence,
                priority_detections=vulnerable_in_danger,
                road_percentage=road_percentage,
                sidewalk_percentage=sidewalk_percentage,
            )

        # 4. CAUTION: Low road visibility
        if road_percentage < self.min_road_ratio:
            return Decision(
                action=Action.CAUTION,
                reason=f"Limited road visibility ({road_percentage:.0%})",
                confidence=0.7,
                priority_detections=[],
                road_percentage=road_percentage,
                sidewalk_percentage=sidewalk_percentage,
            )

        # 5. MAINTAIN: Clear path
        return Decision(
            action=Action.MAINTAIN,
            reason="Path clear - maintain speed",
            confidence=0.9,
            priority_detections=[],
            road_percentage=road_percentage,
            sidewalk_percentage=sidewalk_percentage,
        )

    def detections_from_model_output(
        self,
        model_detections: List[dict],
    ) -> List[Detection]:
        """Convert model output format to Detection objects.

        Args:
            model_detections: List of dicts with 'box', 'class', 'score'

        Returns:
            List of Detection objects
        """
        detections = []
        for det in model_detections:
            detections.append(Detection(
                class_name=self.CLASS_NAMES[det['class']],
                class_id=det['class'],
                box=tuple(det['box']),
                confidence=det['score'],
            ))
        return detections


def test_decision_engine():
    """Test the decision engine."""
    print("Testing Decision Engine...")

    engine = DecisionEngine()

    # Test 1: Person in danger zone
    detections = [
        Detection("person", 0, (400, 200, 500, 450), 0.85),
    ]
    seg_mask = np.ones((512, 1024), dtype=np.uint8)  # All road

    decision = engine.analyze(detections, seg_mask)
    print(f"\nTest 1 - Person ahead:")
    print(f"  Action: {decision.action.value}")
    print(f"  Reason: {decision.reason}")

    # Test 2: Car ahead
    detections = [
        Detection("car", 2, (350, 250, 650, 480), 0.92),
    ]
    decision = engine.analyze(detections, seg_mask)
    print(f"\nTest 2 - Car ahead:")
    print(f"  Action: {decision.action.value}")
    print(f"  Reason: {decision.reason}")

    # Test 3: Clear path
    detections = []
    decision = engine.analyze(detections, seg_mask)
    print(f"\nTest 3 - Clear path:")
    print(f"  Action: {decision.action.value}")
    print(f"  Reason: {decision.reason}")

    print("\n✅ Decision Engine tests passed!")


if __name__ == "__main__":
    test_decision_engine()
