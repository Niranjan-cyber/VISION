import unittest
from src.core.types import BoundingBox
from src.events.zone import Zone, point_in_zone


class TestEventsGeometry(unittest.TestCase):
    """Unit tests for spatial geometry and point-in-polygon evaluation."""

    def setUp(self):
        # 100x100 square from (10, 10) to (110, 110)
        self.square_zone = Zone(
            id="square_zone",
            name="Test Square Zone",
            zone_type="restricted",
            polygon=[(10, 10), (110, 10), (110, 110), (10, 110)],
        )

        # L-shaped (concave) polygon
        self.concave_zone = Zone(
            id="concave_zone",
            name="Test Concave Zone",
            zone_type="restricted",
            polygon=[
                (0, 0),
                (100, 0),
                (100, 50),
                (50, 50),
                (50, 100),
                (0, 100),
            ],
        )

    def test_1_point_strictly_inside(self):
        """Point clearly inside polygon boundaries returns True."""
        self.assertTrue(point_in_zone((50, 50), self.square_zone))
        self.assertTrue(point_in_zone((20, 20), self.square_zone))
        self.assertTrue(point_in_zone((100, 100), self.square_zone))

    def test_2_point_strictly_outside(self):
        """Point outside polygon boundaries returns False."""
        self.assertFalse(point_in_zone((5, 5), self.square_zone))
        self.assertFalse(point_in_zone((120, 50), self.square_zone))
        self.assertFalse(point_in_zone((50, 120), self.square_zone))
        self.assertFalse(point_in_zone((0, 0), self.square_zone))

    def test_3_point_on_boundary(self):
        """Point located exactly on an edge or vertex returns True."""
        # Vertex
        self.assertTrue(point_in_zone((10, 10), self.square_zone))
        # Edge
        self.assertTrue(point_in_zone((50, 10), self.square_zone))
        self.assertTrue(point_in_zone((110, 60), self.square_zone))

    def test_4_concave_polygon_internal_notch(self):
        """Points inside the outer bounds but within a concave notch evaluate as outside."""
        # Inside the L-shape solid part
        self.assertTrue(point_in_zone((25, 25), self.concave_zone))
        self.assertTrue(point_in_zone((25, 75), self.concave_zone))
        self.assertTrue(point_in_zone((75, 25), self.concave_zone))

        # In the notch (x=75, y=75) where polygon cut out
        self.assertFalse(point_in_zone((75, 75), self.concave_zone))

    def test_5_bottom_center_ground_position_evaluation(self):
        """BoundingBox bottom_center ground position correctly tests zone membership."""
        # Person standing inside square: feet at (50, 100)
        box_inside = BoundingBox(x1=30, y1=20, x2=70, y2=100)
        self.assertEqual(box_inside.bottom_center, (50, 100))
        self.assertTrue(point_in_zone(box_inside.bottom_center, self.square_zone))

        # Person standing just outside bottom: feet at (50, 115)
        box_outside = BoundingBox(x1=30, y1=35, x2=70, y2=115)
        self.assertEqual(box_outside.bottom_center, (50, 115))
        self.assertFalse(point_in_zone(box_outside.bottom_center, self.square_zone))


if __name__ == "__main__":
    unittest.main()
