from trie import HexaryTrie

class AccountManager:
    def __init__(self):
        # Initializing a blank trie
        self.trie = HexaryTrie(db={})
        self.block_data = {}  # To store root hash of trie for each block

    def set_balance(self, account, block, amount):
        self.trie[account.encode()] = str(amount).encode()
        # Save the root hash of the trie for the given block
        self.block_data[block] = self.trie.root_hash

    def get_balance(self, account, block):
        root = self.block_data.get(block)
        if not root:
            return None
        # Set the trie's root hash to the one from the desired block
        self.trie.root_hash = root
        return int(self.trie[account.encode()].decode())

    def credit(self, account, block, amount):
        curr_balance = self.get_balance(account, block-1) or 0
        self.set_balance(account, block, curr_balance + amount)

    def debit(self, account, block, amount):
        curr_balance = self.get_balance(account, block-1) or 0
        if curr_balance < amount:
            raise ValueError("Insufficient balance")
        self.set_balance(account, block, curr_balance - amount)

if __name__ == "__main__":
    manager = AccountManager()

    # Test setting initial balance
    manager.set_balance('0xSomeAccount', 1, 100)
    assert manager.get_balance('0xSomeAccount', 1) == 100

    # Test crediting an account
    manager.credit('0xSomeAccount', 2, 50)
    assert manager.get_balance('0xSomeAccount', 1) == 100
    assert manager.get_balance('0xSomeAccount', 2) == 150

    # Test debiting an account
    manager.debit('0xSomeAccount', 3, 30)
    assert manager.get_balance('0xSomeAccount', 1) == 100
    assert manager.get_balance('0xSomeAccount', 2) == 150
    assert manager.get_balance('0xSomeAccount', 3) == 120

    print("All tests passed!")

