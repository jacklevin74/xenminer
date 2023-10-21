import re
import base64
from web3 import Web3

def restore_eip55_address(lowercase_address: str) -> bool:
    # Restore the address using EIP-55 checksum
    try:
        checksum_address = Web3.to_checksum_address(lowercase_address)
        print("Checksummed address is: ", checksum_address)
        return True
    except ValueError as e:
        # Handle the error in case the address is not a valid Ethereum address
        print(f"An error occurred: {e}")
        return False


def check_salt_format_and_ethereum_address(hash_to_verify: str) -> bool:
    # Regular expressions for the expected patterns
    pattern1 = re.compile(r'WEVOMTAwODIwMjJYRU4')
    pattern2 = re.compile(r'^[A-Za-z0-9+/]{27}$')

    # Extract the salt part from the hash_to_verify
    parts = hash_to_verify.split("$")
    if len(parts) != 6:
        return False
    salt = parts[4]

    # Check if the salt matches the first pattern
    if pattern1.search(salt):
        return True
    else:
        print("Old Salt False")

    # Check if the salt matches the second pattern and is base64
    if pattern2.fullmatch(salt):
        print ("In Pattern2 match")
        try:
            # The proper base64 string should have a length that is a multiple of 4.
            # We need to add padding if necessary.
            missing_padding = len(salt) % 4
            if missing_padding:
                salt += '=' * (4 - missing_padding)

            # Decode the base64 string
            decoded_bytes = base64.b64decode(salt)
            decoded_str = decoded_bytes.hex()
            print("Decoded salt: ", decoded_str)

            # Check if the decoded string is a valid hexadecimal and of a specific length
            if re.fullmatch(r'[0-9a-fA-F]{40}', decoded_str):  # Ethereum addresses are 40 hex characters long
                # Construct potential Ethereum address
                potential_eth_address = '0x' + decoded_str
                print("Address matched: ", potential_eth_address)

                # Validate Ethereum address checksum
                if restore_eip55_address(potential_eth_address):
                    print("Checksum of address is valid")
                    return True
        except Exception as e:
            print(f"An error occurred: {e}")
            return False

    return False

# Example for testing the function
hash_to_verify = "$argon2id$v=19$m=1500,t=1,p=1$pyWn54uTptICzki0Oa5DknaaiqM$SFjWpTvU0PwEr+Y56fLhFIRL4FWhHXENqJ48RUr1ipApiovrb/ciR5D5//etMI7unr/133U8PKQ6mBWoR39few"
print(check_salt_format_and_ethereum_address(hash_to_verify))  # True if it's a valid Ethereum address with a valid checksum
hash_to_verify = "$argon2id$v=19$m=67400,t=1,p=1$WEVOMTAwODIwMjJYRU4$CiW038mdwhkuhBQHEsUr+08ljPAXUNI7eRyIvkvXnsJELQHyCS0wCxyH3oHjBz1ymjBX34jTN30z+id6Ap1xDw"
print(check_salt_format_and_ethereum_address(hash_to_verify))  # True if it's a valid Ethereum address with a valid checksum
hash_to_verify = "$argon2id$v=19$m=1500,t=1,p=1$QkShdDvVej2+KqeRSqCih0N6Dvs$At03JjGusrxCib4ejiGJ6Boziavjp9l7uWF+2Hb1CWmvXENxUNlBCxAXW8nBmW466h4iZzjYgNuiEtytCF+roA"
print(check_salt_format_and_ethereum_address(hash_to_verify))  # True if it's a valid Ethereum address with a valid checksum

