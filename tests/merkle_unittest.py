import unittest
import hashlib

def hash_value(value):
    return hashlib.sha256(value.encode()).hexdigest()

def build_merkle_tree(elements, merkle_tree=None):
    if merkle_tree is None:
        merkle_tree = {}
    if len(elements) == 1:
        return elements[0], merkle_tree

    new_elements = []
    for i in range(0, len(elements), 2):
        left = elements[i]
        right = elements[i + 1] if i + 1 < len(elements) else left
        combined = left + right
        new_hash = hash_value(combined)
        merkle_tree[new_hash] = {'left': left, 'right': right}
        new_elements.append(new_hash)
    return build_merkle_tree(new_elements, merkle_tree)

class TestMerkleTreeFunctions(unittest.TestCase):
    
    def test_hash_value(self):
        self.assertEqual(hash_value('hello'), hashlib.sha256('hello'.encode()).hexdigest())

    def test_build_merkle_tree(self):
        elements = ['a', 'b', 'c', 'd']
        root, tree = build_merkle_tree(elements)
        print ("Merkleroot hash: ", root)

        # Test if the Merkle tree has the correct number of nodes
        self.assertEqual(len(tree), len(elements) - 1)

        # Test if the root is correct. This is a simple check and assumes
        # you know what the root should be for the elements ['a', 'b', 'c', 'd'].
        expected_root = hash_value(hash_value('a' + 'b') + hash_value('c' + 'd'))
        self.assertEqual(root, expected_root)

        # Add more tests as necessary to cover the logic of your Merkle tree construction.

if __name__ == '__main__':
    unittest.main()

