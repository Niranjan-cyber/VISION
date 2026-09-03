import os
import tempfile
import unittest
import yaml

from src.events.zone import (
    Zone,
    load_zones_from_dict,
    load_zones_from_file,
)


class TestEventsZoneConfig(unittest.TestCase):
    """Unit test suite for zone configuration loading and schema validation."""

    def test_1_valid_dict_loading(self):
        """Valid dictionary with multiple zones parses correctly."""
        data = {
            "zones": [
                {
                    "id": "zone_a",
                    "name": "Perimeter Alpha",
                    "type": "restricted",
                    "polygon": [(0, 0), (100, 0), (100, 100), (0, 100)],
                    "metadata": {"floor": 1},
                },
                {
                    "id": "zone_b",
                    "name": "Perimeter Beta",
                    "type": "warning",
                    "polygon": [(200, 200), (300, 200), (250, 300)],
                },
            ]
        }
        zones = load_zones_from_dict(data)
        self.assertEqual(len(zones), 2)
        self.assertEqual(zones[0].id, "zone_a")
        self.assertEqual(zones[0].zone_type, "restricted")
        self.assertEqual(len(zones[0].polygon), 4)
        self.assertEqual(zones[1].id, "zone_b")
        self.assertEqual(zones[1].zone_type, "warning")
        self.assertEqual(len(zones[1].polygon), 3)

    def test_2_valid_yaml_file_loading(self):
        """Valid YAML configuration file loads correctly."""
        data = {
            "zones": [
                {
                    "id": "bop_zone",
                    "name": "BOP Gate",
                    "type": "restricted",
                    "polygon": [[50, 50], [250, 50], [250, 250], [50, 250]],
                }
            ]
        }
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            yaml.dump(data, f)
            temp_path = f.name

        try:
            zones = load_zones_from_file(temp_path)
            self.assertEqual(len(zones), 1)
            self.assertEqual(zones[0].id, "bop_zone")
            self.assertEqual(zones[0].polygon, [(50, 50), (250, 50), (250, 250), (50, 250)])
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_3_missing_zones_key_raises_error(self):
        """Missing 'zones' key raises ValueError."""
        with self.assertRaises(ValueError):
            load_zones_from_dict({"not_zones": []})

    def test_4_empty_zones_list_raises_error(self):
        """Empty list under 'zones' raises ValueError."""
        with self.assertRaises(ValueError):
            load_zones_from_dict({"zones": []})

    def test_5_duplicate_zone_id_raises_error(self):
        """Duplicate zone IDs raise ValueError."""
        data = {
            "zones": [
                {
                    "id": "dup_zone",
                    "name": "First Instance",
                    "type": "restricted",
                    "polygon": [(0, 0), (10, 0), (10, 10)],
                },
                {
                    "id": "dup_zone",
                    "name": "Second Instance",
                    "type": "warning",
                    "polygon": [(20, 20), (30, 20), (30, 30)],
                },
            ]
        }
        with self.assertRaises(ValueError) as ctx:
            load_zones_from_dict(data)
        self.assertIn("Duplicate zone ID", str(ctx.exception))

    def test_6_fewer_than_3_vertices_raises_error(self):
        """Zone with less than 3 points is an invalid polygon."""
        with self.assertRaises(ValueError):
            Zone(id="invalid_poly", name="Test", zone_type="restricted", polygon=[(0, 0), (10, 10)])

    def test_7_malformed_coordinates_raises_error(self):
        """Non-numeric or non-2D points raise ValueError."""
        with self.assertRaises(ValueError):
            Zone(id="bad_coords", name="Test", zone_type="restricted", polygon=[(0, 0), ("bad", 10), (10, 10)])

        with self.assertRaises(ValueError):
            Zone(id="bad_tuple", name="Test", zone_type="restricted", polygon=[(0, 0), (10, 10, 10), (20, 20)])

    def test_8_invalid_zone_type_raises_error(self):
        """Unsupported zone type raises ValueError."""
        with self.assertRaises(ValueError):
            Zone(id="bad_type", name="Test", zone_type="secret_zone", polygon=[(0, 0), (10, 0), (10, 10)])

    def test_9_missing_file_raises_file_not_found(self):
        """Non-existent file path raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            load_zones_from_file("non_existent_zones_path_12345.yaml")


if __name__ == "__main__":
    unittest.main()
