import unittest
from src.anpr.cleaner import clean_plate_text, disambiguate_indian_plate, is_valid_indian_plate, strip_non_alphanumeric


class TestPlateCleaner(unittest.TestCase):
    def test_strip_non_alphanumeric(self):
        self.assertEqual(strip_non_alphanumeric("MH-12-AB-1234"), "MH12AB1234")
        self.assertEqual(strip_non_alphanumeric("DL 01 A 9999"), "DL01A9999")
        self.assertEqual(strip_non_alphanumeric("KA.05:MN-1234!"), "KA05MN1234")
        self.assertEqual(strip_non_alphanumeric(""), "")
        self.assertEqual(strip_non_alphanumeric(None), "")

    def test_valid_indian_plates(self):
        self.assertTrue(is_valid_indian_plate("MH12AB1234"))
        self.assertTrue(is_valid_indian_plate("DL01A9999"))
        self.assertTrue(is_valid_indian_plate("KA05MN5678"))
        self.assertTrue(is_valid_indian_plate("22BH1234AA"))
        # Invalid state code
        self.assertFalse(is_valid_indian_plate("ZZ12AB1234"))
        # Invalid length
        self.assertFalse(is_valid_indian_plate("MH12"))

    def test_disambiguation(self):
        # 'O' in digit position corrected to '0'
        self.assertEqual(disambiguate_indian_plate("MHO2AB1234"), "MH02AB1234")
        # '8' in letter position corrected to 'B'
        self.assertEqual(disambiguate_indian_plate("MH12A81234"), "MH12AB1234")
        # '0' in state prefix corrected to 'O' (e.g. Odisha OD)
        self.assertEqual(disambiguate_indian_plate("0D02AB1234"), "OD02AB1234")
        # 'I' in number position corrected to '1'
        self.assertEqual(disambiguate_indian_plate("MH12ABI234"), "MH12AB1234")

    def test_clean_plate_text_end_to_end(self):
        # Direct valid plate
        cleaned, is_valid, score = clean_plate_text("MH 12 AB 1234")
        self.assertEqual(cleaned, "MH12AB1234")
        self.assertTrue(is_valid)
        self.assertEqual(score, 1.0)

        # Plate with OCR confusion
        cleaned, is_valid, score = clean_plate_text("MH-O2-A8-I234")
        self.assertEqual(cleaned, "MH02AB1234")
        self.assertTrue(is_valid)
        self.assertEqual(score, 1.0)

        # Generic alphanumeric plate
        cleaned, is_valid, score = clean_plate_text("ABC1234")
        self.assertEqual(cleaned, "ABC1234")
        self.assertTrue(is_valid)
        self.assertEqual(score, 0.80)

        # Empty
        cleaned, is_valid, score = clean_plate_text("")
        self.assertEqual(cleaned, "")
        self.assertFalse(is_valid)

    def test_single_letter_prefix_expansion(self):
        # 8-char OCR where IND stripe corrupted M into W or H
        self.assertEqual(disambiguate_indian_plate("W50U7737"), "MH50U7737")
        self.assertEqual(disambiguate_indian_plate("H50U1737"), "MH50U1737")
        self.assertEqual(disambiguate_indian_plate("M50U7737"), "MH50U7737")

    def test_format_indian_plate(self):
        from src.anpr.cleaner import format_indian_plate
        self.assertEqual(format_indian_plate("MH50U7737"), "MH 50 U 7737")
        self.assertEqual(format_indian_plate("MH12AB1234"), "MH 12 AB 1234")
        self.assertEqual(format_indian_plate("ABC1234"), "ABC1234")
        self.assertEqual(format_indian_plate(""), "")


if __name__ == "__main__":
    unittest.main()
