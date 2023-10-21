package main

import (
 "encoding/hex"
 "fmt"
 "log"
 "time"

 "github.com/ethereum/go-ethereum/core/types"
 "github.com/ethereum/go-ethereum/common"
 "github.com/ethereum/go-ethereum/crypto"
 "github.com/ethereum/go-ethereum/rlp"
)

func getRecoveredAddress(rawTx string) *string {
 // Decode the input string to bytes
 rawTxBytes, err := hex.DecodeString(rawTx[2:]) // remove 0x prefix
 if err != nil {
  log.Fatalf("Invalid transaction hex: %v", err)
 }

 // Parse the transaction
 tx := new(types.Transaction)
 err = rlp.DecodeBytes(rawTxBytes, tx) // Use RLP to decode bytes into the transaction structure
 if err != nil {
  log.Fatalf("Error unmarshalling transaction: %v", err)
 }

 // Get chain ID and create a signer
 chainID := tx.ChainId()
 signer := types.NewEIP155Signer(chainID)

 // Get the transaction hash using the signer
 sigHash := signer.Hash(tx)

 // Extract the signature and recover the sender's address
 V, R, S := tx.RawSignatureValues()

 // Adjust V for EIP-155
 var V_dec uint64
 if chainID.Sign() != 0 {
  V_dec = V.Uint64() - uint64(2*chainID.Uint64()+8) // for EIP-155
 } else {
  V_dec = V.Uint64() // for non-EIP-155, no adjustment needed
 }

 // Check if V is 27 or 28, adjust to 0 or 1 for compatibility with Ecrecover
 if V_dec == 27 || V_dec == 28 {
  V_dec -= 27
 }

 if V_dec > 1 {
  log.Fatalf("Invalid V value: %v", V_dec)
 }

 // Construct the public key from the signature
 sig := append(R.Bytes(), S.Bytes()...)          // R || S
 sig = append(sig, byte(V_dec))                 // R || S || V_dec
 pubKey, err := crypto.Ecrecover(sigHash[:], sig)
 if err != nil {
  log.Fatalf("Error recovering public key: %v", err)
 }

 var addr common.Address
 copy(addr[:], crypto.Keccak256(pubKey[1:])[12:]) // take the last 20 bytes

 address := addr.Hex()
 return &address
}

func benchmarkDecode(rawTx string, numTrials int) []time.Duration {
 executionTimes := make([]time.Duration, 0, numTrials)

 for i := 0; i < numTrials; i++ {
  startTime := time.Now()

  address := getRecoveredAddress(rawTx)
  if address == nil {
   log.Println("Error getting recovered address")
   continue
  }

  endTime := time.Now()
  executionTimes = append(executionTimes, endTime.Sub(startTime))
 }

 return executionTimes
}

func computeMetrics(executionTimes []time.Duration) (avgTime time.Duration, txPerSecond float64) {
 var total time.Duration
 for _, execTime := range executionTimes {
  total += execTime
 }

 avgTime = total / time.Duration(len(executionTimes))
 txPerSecond = float64(time.Second) / float64(avgTime)
 return
}

func main() {
 // This is your raw Ethereum transaction.
 rawTx := "0xf8ab80843b9aca00827b0c94999999cf1046e68e36e1aa2e0e07105eddd0000280b844a9059cbb000000000000000000000000f9b8dd0565b1fac5e9a142c3553f663e09444b9c000000000000000000000000000000000000000000000001236efcbcbb34000083030e2da0919b7300531a90a5e36a54d12e59cb33c2f77a3ea340f92bad063030c209bff1a059ca1d78f3ed0b017e07a6214263fb8a4433a7feff7ac4f0dbe771c1130beaf5"
 numTrials := 100000

 executionTimes := benchmarkDecode(rawTx, numTrials)
 avgTime, txPerSecond := computeMetrics(executionTimes)

 // Output the metrics
 fmt.Printf("\nMetrics over %d trials:\n", numTrials)
 fmt.Printf("Average Time: %v\n", avgTime)
 fmt.Printf("Transactions per Second: %.2f\n", txPerSecond)
 // Recover and print the address
 recoveredAddress := getRecoveredAddress(rawTx)
 fmt.Printf("Recovered Address: %s\n", *recoveredAddress)
}
