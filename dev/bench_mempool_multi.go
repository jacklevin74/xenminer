package main

import (
	"bytes"
	"context"
	"encoding/hex"
	"fmt"
	"log"
	"runtime"
	"sync"
	"time"

	"github.com/ethereum/go-ethereum/core/types"
	"github.com/ethereum/go-ethereum/rlp"
	"github.com/go-redis/redis/v8"
)

var ctx = context.Background()

func getRecoveredAddress(rawTx string) *string {
	rawTxBytes, err := hex.DecodeString(rawTx)
	if err != nil {
		log.Printf("Invalid transaction hex: %v", err)
		return nil
	}

	tx := new(types.Transaction)

	r := bytes.NewReader(rawTxBytes)
	if err := rlp.Decode(r, tx); err != nil {
		log.Printf("Error decoding transaction: %v", err)
		return nil
	}

	chainID := tx.ChainId()
	signer := types.NewEIP155Signer(chainID)
	senderAddr, err := types.Sender(signer, tx)
	if err != nil {
		log.Printf("Error extracting sender address from transaction: %v", err)
		return nil
	}

	senderHex := senderAddr.Hex()
	return &senderHex
}

func worker(wg *sync.WaitGroup, semaphore chan struct{}, rdb *redis.Client, key string, rawTxHex string) {
	defer wg.Done()

	rawTxBytes, err := hex.DecodeString(rawTxHex) // remove 0x prefix
	if err != nil {
		log.Printf("Invalid transaction hex: %v", err)
		return
	}

	err = rdb.RPush(ctx, key, rawTxBytes).Err() // storing bytes
	if err != nil {
		log.Printf("Error writing to Redis: %v", err)
		return
	}

	// Simulate processing by immediately popping and decoding
	result, err := rdb.LPop(ctx, key).Bytes()
	if err != nil {
		log.Printf("Error reading from Redis: %v", err)
		return
	}

	rawTx := hex.EncodeToString(result)
	recoveredAddress := getRecoveredAddress(rawTx)
	if recoveredAddress == nil {
		log.Println("Error getting recovered address")
		return
	}

	// Release the semaphore
	<-semaphore
}

func writeAndProcessTransactions(rdb *redis.Client, key string, rawTxHex string, count int) {
	start := time.Now()

	numCores := runtime.NumCPU()
	semaphore := make(chan struct{}, numCores)
	var wg sync.WaitGroup

	for i := 0; i < count; i++ {
		wg.Add(1)

		// Acquire the semaphore
		semaphore <- struct{}{}

		go worker(&wg, semaphore, rdb, key, rawTxHex)
	}

	// Wait for all goroutines to complete
	wg.Wait()

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
	numCores := runtime.NumCPU()
	rdb := redis.NewClient(&redis.Options{
		Addr: "127.0.0.1:6379",
		DB:   0, // use default DB
	})

	key := "transactions" // the key where transactions are stored in Redis
	rawTx := "f8ab80843b9aca00827b0c94999999cf1046e68e36e1aa2e0e07105eddd0000280b844a9059cbb000000000000000000000000f9b8dd0565b1fac5e9a142c3553f663e09444b9c000000000000000000000000000000000000000000000001236efcbcbb34000083030e2da0919b7300531a90a5e36a54d12e59cb33c2f77a3ea340f92bad063030c209bff1a059ca1d78f3ed0b017e07a6214263fb8a4433a7feff7ac4f0dbe771c1130beaf5"
	numberOfTransactions := 1000000 // This is a large number, adjust accordingly

	// Ensure maximum parallelism
	runtime.GOMAXPROCS(numCores*2)

	// Simulate write and process transactions
	writeAndProcessTransactions(rdb, key, rawTx, numberOfTransactions)
}
