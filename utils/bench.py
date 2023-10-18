import rlp
from eth_typing import HexStr
from eth_utils import decode_hex
from timeit import default_timer as timer
from statistics import mean
from tqdm import tqdm  # new import

# If you're using Ethereum transactions, you might also need the Transaction class from the 'ethereum' library.
# If you haven't got 'ethereum' installed, you can install it using pip: pip install ethereum
from ethereum.transactions import Transaction

def benchmark_decode(raw_tx: str, num_trials: int = 1000000):
    # Prepare the data
    raw_tx_bytes = bytes.fromhex(HexStr(raw_tx[2:]))

    # Store each trial's execution time
    execution_times = []

    # Using tqdm to create a progress bar for the loop
    for _ in tqdm(range(num_trials), desc="Decoding", unit="tx"):  # wrapped range with tqdm
        start_time = timer()
        decoded_tx = rlp.decode(raw_tx_bytes, Transaction)
        end_time = timer()
        execution_times.append(end_time - start_time)

    return execution_times

def compute_metrics(execution_times):
    avg_time = mean(execution_times)

    # Calculate transactions per second
    tx_per_second = 1 / avg_time if avg_time > 0 else 0

    return {
        'avg_time': avg_time,
        'tx_per_second': tx_per_second
    }

def main():
    # This is your raw Ethereum transaction.
    raw_tx = "0xf8ab80843b9aca00827b0c94999999cf1046e68e36e1aa2e0e07105eddd0000280b844a9059cbb000000000000000000000000f9b8dd0565b1fac5e9a142c3553f663e09444b9c000000000000000000000000000000000000000000000001236efcbcbb34000083030e2da0919b7300531a90a5e36a54d12e59cb33c2f77a3ea340f92bad063030c209bff1a059ca1d78f3ed0b017e07a6214263fb8a4433a7feff7ac4f0dbe771c1130beaf5"
    num_trials = 1000000
    raw_tx_bytes = bytes.fromhex(HexStr(raw_tx[2:]))
    decoded_tx = rlp.decode(raw_tx_bytes, Transaction)
    from_account = decoded_tx.to
    print(from_account.hex())

    execution_times = benchmark_decode(raw_tx, num_trials)
    metrics = compute_metrics(execution_times)

    # Output the metrics
    print(f"\nMetrics over {num_trials} trials:")  # added a newline for cleaner output
    print(f"Average Time: {metrics['avg_time']:.6f} seconds")
    print(f"Transactions per Second: {metrics['tx_per_second']:.2f}")

if __name__ == "__main__":
    main()

