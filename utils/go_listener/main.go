package main

import (
	"log"
	"strings"
	"time"
	"strconv"
	"encoding/json"
	"crypto/sha256"
	"encoding/hex"
	"fmt"

	"github.com/gorilla/websocket"
)

func main() {
	var addr = "ws://xenblocks.io:6668"
	c, _, err := websocket.DefaultDialer.Dial(addr, nil)
	if err != nil {
		log.Fatal("dial:", err)
	}
	defer c.Close()

	fmt.Println("Connected to the server!")

	for {
		_, message, err := c.ReadMessage()
		if err != nil {
			log.Println("read:", err)
			time.Sleep(5 * time.Second)
			continue
		}

		decompressedData, err := decompressData(message)
		if err != nil {
			log.Printf("decompress error: %v", err)
			continue
		}

		parts := strings.Split(decompressedData, "|")
		if len(parts) < 6 {
			continue // Not enough parts, skip this message
		}

		blockID := parts[0]
		hashToVerify := parts[1]
		key := parts[2]

		// Start the performance timer
		start := time.Now()
		var truncatedHash = ""
		var bool_flag = false

		// Perform Argon2 verification for the single block ID
		if verifyArgon2Hash(hashToVerify, key) {
			var shaHash = sha256.Sum256([]byte(hashToVerify))
			// Convert the SHA256 hash to a hexadecimal string and take the first 5 characters
			truncatedHash = hex.EncodeToString(shaHash[:])[:5]
			bool_flag = true
			//fmt.Printf("Verification success for block ID %s\n", blockID)
		} else {
			fmt.Printf("Verification failed for block ID %s\n", blockID)
			continue
		}
		// Stop the performance timer and print the elapsed time
		elapsed := time.Since(start)
		elapsedInMilliseconds := elapsed / time.Millisecond
		timeStr := strconv.FormatInt(int64(elapsedInMilliseconds), 10)

		fmt.Printf("Verification time for block ID %s: %s %s\n", blockID,truncatedHash,elapsed)

		if bool_flag {

			responseData := map[string]interface{}{
				"block_id":   blockID,
				"hash":       truncatedHash,
				"time_diff":  timeStr,
			}

			// Marshal the response data to JSON
			jsonData, err := json.Marshal(responseData)
			if err != nil {
				log.Printf("JSON marshal error: %v", err)
				continue
			}

			// Send the JSON response back through the WebSocket
			if err := c.WriteMessage(websocket.TextMessage, jsonData); err != nil {
				log.Printf("write error: %v", err)
				continue
			}
		}

	}
}

