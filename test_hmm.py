import unittest
import tempfile
import os

from HMM import HMM


# used Claude to help me create unit tests for the load function

class TestHMM(unittest.TestCase):
    def setUp(self):
        # Create temporary directory for test files
        self.test_dir = tempfile.mkdtemp()

        # Create test transition file content
        trans_content = """# happy 0.5
# grumpy 0.5
# hungry 0
happy happy 0.5
happy grumpy 0.1
happy hungry 0.4
grumpy happy 0.6
grumpy grumpy 0.3
grumpy hungry 0.1
hungry happy 0.1
hungry grumpy 0.6
hungry hungry 0.3"""

        # Create test emission file content
        emit_content = """happy silent 0.2
happy meow 0.3
happy purr 0.5
grumpy silent 0.5
grumpy meow 0.4
grumpy purr 0.1
hungry silent 0.2
hungry meow 0.6
hungry purr 0.2"""

        # Write test files
        self.basename = os.path.join(self.test_dir, "test_cat")
        with open(f"{self.basename}.trans", "w") as f:
            f.write(trans_content)
        with open(f"{self.basename}.emit", "w") as f:
            f.write(emit_content)

    def tearDown(self):
        # Clean up test files
        os.remove(f"{self.basename}.trans")
        os.remove(f"{self.basename}.emit")
        os.rmdir(self.test_dir)

    def test_load_transitions(self):
        hmm = HMM()
        hmm.load(self.basename)

        expected_transitions = {
            '#': {'happy': '0.5', 'grumpy': '0.5', 'hungry': '0'},
            'happy': {'happy': '0.5', 'grumpy': '0.1', 'hungry': '0.4'},
            'grumpy': {'happy': '0.6', 'grumpy': '0.3', 'hungry': '0.1'},
            'hungry': {'happy': '0.1', 'grumpy': '0.6', 'hungry': '0.3'}
        }

        self.assertEqual(hmm.transitions, expected_transitions,
                         "Transitions were not loaded correctly")

    def test_load_emissions(self):
        hmm = HMM()
        hmm.load(self.basename)

        expected_emissions = {
            'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
            'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
            'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}
        }

        self.assertEqual(hmm.emissions, expected_emissions,
                         "Emissions were not loaded correctly")

    def test_load_nonexistent_file(self):
        hmm = HMM()
        with self.assertRaises(FileNotFoundError):
            hmm.load("nonexistent_file")

    def test_empty_files(self):
        # Create empty files
        empty_basename = os.path.join(self.test_dir, "empty_test")
        with open(f"{empty_basename}.trans", "w") as f:
            pass
        with open(f"{empty_basename}.emit", "w") as f:
            pass

        hmm = HMM()
        hmm.load(empty_basename)

        self.assertEqual(hmm.transitions, {},
                         "Transitions should be empty for empty file")
        self.assertEqual(hmm.emissions, {},
                         "Emissions should be empty for empty file")

        os.remove(f"{empty_basename}.trans")
        os.remove(f"{empty_basename}.emit")


def run_tests():
    print()
    print("Running tests...")
    unittest.main()