package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"fmt"
	"log"
	"time"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/rlp"
	"github.com/go-redis/redis/v8"
)

var ctx = context.Background()

func getRecoveredAddress(rawTx string) *string {
	rawTxBytes, err := hex.DecodeString(rawTx)
	if err != nil {
		log.Fatalf("Invalid transaction hex: %v", err)
	}

	tx := new(types.Transaction)

	r := bytes.NewReader(rawTxBytes)
	if err := rlp.Decode(r, tx); err != nil {
		log.Fatalf("Error decoding transaction: %v", err)
	}

	chainID := tx.ChainId()
	signer := types.NewEIP155Signer(chainID)
	senderAddr, err := types.Sender(signer, tx)
	if err != nil {
		log.Fatalf("Error extracting sender address from transaction: %v", err)
	}

	senderHex := senderAddr.Hex()
	return &senderHex
}

func writeAndProcessTransactions(rdb *redis.Client, key string, rawTxHex string, count int) {
	start := time.Now()

	rawTxBytes, err := hex.DecodeString(rawTxHex) // remove 0x prefix
	if err != nil {
		log.Fatalf("Invalid transaction hex: %v", err)
	}

	for i := 0; i < count; i++ {
		err := rdb.RPush(ctx, key, rawTxBytes).Err() // storing bytes
		if err != nil {
			log.Fatalf("Error writing to Redis: %v", err)
		}

		// Simulate processing by immediately popping and decoding
		result, err := rdb.LPop(ctx, key).Bytes()
		if err != nil {
			log.Fatalf("Error reading from Redis: %v", err)
		}

		rawTx := hex.EncodeToString(result)
		recoveredAddress := getRecoveredAddress(rawTx)
		if recoveredAddress == nil {
			log.Println("Error getting recovered address")
			continue
		}
	}

	end := time.Now()
	elapsed := end.Sub(start)

	fmt.Printf("Wrote and processed %d transactions in %v\n", count, elapsed)

	if count > 0 {
		avgTimePerTx := elapsed / time.Duration(count)
		txPerSecond := float64(count) / elapsed.Seconds()

		fmt.Printf("Average time per transaction: %v\n", avgTimePerTx)
		fmt.Printf("Transactions per second: %v\n", txPerSecond)
	}
}

func main() {
	rdb := redis.NewClient(&redis.Options{
		Addr: "127.0.0.1:6379",
		DB:   0, // use default DB
	})

	key := "transactions" // the key where transactions are stored in Redis
	rawTx := "f8ab80843b9aca00827b0c94999999cf1046e68e36e1aa2e0e07105eddd0000280b844a9059cbb000000000000000000000000f9b8dd0565b1fac5e9a142c3553f663e09444b9c000000000000000000000000000000000000000000000001236efcbcbb34000083030e2da0919b7300531a90a5e36a54d12e59cb33c2f77a3ea340f92bad063030c209bff1a059ca1d78f3ed0b017e07a6214263fb8a4433a7feff7ac4f0dbe771c1130beaf5"
	numberOfTransactions := 500000 // This is a large number, adjust accordingly

	// Simulate write and process transactions
	writeAndProcessTransactions(rdb, key, rawTx, numberOfTransactions)
}
