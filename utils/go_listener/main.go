package main

import (
	"strings"
	"time"
	"strconv"
	"encoding/json"
	"os"
	"crypto/sha256"
	"runtime"
	"encoding/hex"
	"fmt"
	"flag"
	"context"
	"github.com/libp2p/go-libp2p-pubsub"
	"github.com/gorilla/websocket"
        "net/http"
        "net/url"
	"github.com/libp2p/go-libp2p"
//	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	log "github.com/sirupsen/logrus"
//	log2 "github.com/ipfs/go-log/v2"
	"github.com/multiformats/go-multiaddr"
)

func reportGCTime() {
	for {
		var m runtime.MemStats
		runtime.ReadMemStats(&m)
		fmt.Printf("Total GC time: %v ms\n", m.PauseTotalNs/1e6)

		// Sleep for 5 seconds before reporting again.
		time.Sleep(1 * time.Second)
	}
}

func verifyArgon2HashRemotely(hashToVerify, key string) (bool, string, error) {
    // Construct the request URL
    verifyURL := "http://209.124.84.6:5011/verify"
    //verifyURL := "http://localhost:5011/verify"

    // Create form data
    formData := url.Values{}
    formData.Set("hashToVerify", hashToVerify)
    formData.Set("key", key)

    // Create a new request
    resp, err := http.PostForm(verifyURL, formData)
    if err != nil {
        // Handle error
        return false, "", err
    }
    defer resp.Body.Close()

    // Check the response code
    if resp.StatusCode != http.StatusOK {
        // Verification failed
        return false, "", nil
    }

    // If verification succeeded, compute the truncated hash
    shaHash := sha256.Sum256([]byte(hashToVerify))
    truncatedHash := hex.EncodeToString(shaHash[:])[:5]

    // Return the results
    return true, truncatedHash, nil
}

// handleStream is the stream handler for incoming streams.
// It should contain the logic to handle the incoming stream data.
func handleStream(stream network.Stream) {
	// Stream handler logic goes here.
	fmt.Println("New stream initiated.")
	// Remember to close the stream when done.
	defer stream.Close()
}

func printMemStats() {
        var m runtime.MemStats

        for {
                runtime.ReadMemStats(&m)

                fmt.Printf("Alloc = %v MiB", bToMb(m.Alloc))
                fmt.Printf("\tTotalAlloc = %v MiB", bToMb(m.TotalAlloc))
                fmt.Printf("\tSys = %v MiB", bToMb(m.Sys))
                fmt.Printf("\tNumGC = %v\n", m.NumGC)

                time.Sleep(1 * time.Second)
        }
}

func bToMb(b uint64) uint64 {
        return b / 1024 / 1024
}


func main() {

	// log2.SetAllLoggers(log2.LevelDebug)
	go printMemStats()
	go reportGCTime()

        // Accept the bootstrap node's multiaddress from the command line.
	bootstrapAddr := flag.String("bootstrap", "", "The multiaddress of the bootstrap node")
	flag.Parse()

        // Setup context and host
	ctx := context.Background()
	h, err := libp2p.New()
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create host: %v\n", err)
		os.Exit(1)
	}
	defer h.Close()

	h.SetStreamHandler("/ipfs/kad/1.0.0", handleStream)

        // If the bootstrap flag is provided, parse the multiaddress
	// and connect to the bootstrap node.
	if *bootstrapAddr != "" {
		bootstrapMA, err := multiaddr.NewMultiaddr(*bootstrapAddr)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Invalid bootstrap multiaddress: %v\n", err)
			os.Exit(1)
		}

		bootstrapAI, err := peer.AddrInfoFromP2pAddr(bootstrapMA)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error parsing bootstrap multiaddress: %v\n", err)
			os.Exit(1)
		}

		if err := h.Connect(ctx, *bootstrapAI); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to connect to bootstrap node: %v\n", err)
			os.Exit(1)
		}

		fmt.Println("Successfully connected to bootstrap node:", *bootstrapAddr)
	}

	var dhtOptions []dht.Option
	dhtOptions = append(dhtOptions, dht.Mode(dht.ModeServer))

        kadDHT, err := dht.New(ctx, h, dhtOptions...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create DHT: %v\n", err)
		os.Exit(1)
	}

	if *bootstrapAddr == "" {
		if err = kadDHT.Bootstrap(ctx); err != nil {
			fmt.Fprintf(os.Stderr, "Failed to bootstrap DHT: %v\n", err)
			os.Exit(1)
		}
	}

    // Print the host's full multiaddresses
    addrs := h.Addrs()
    id := h.ID()
    for _, addr := range addrs {
        fullAddr := fmt.Sprintf("%s/p2p/%s", addr, id)
	log.Infof("Full multiaddress:", fullAddr)
    }


    	ps, err := pubsub.NewGossipSub(ctx, h)
    	if err != nil {
        	log.Fatalf("Failed to create pubsub: %v", err)
    	}

	    subscriptionTopic, err := ps.Join("your-topic-name")
	    if err != nil {
		log.Fatalf("Failed to join topic: %v", err)
	    }
	    defer subscriptionTopic.Close()

	   sub, err := subscriptionTopic.Subscribe()
		if err != nil {
			log.Fatalf("Failed to subscribe to subscription topic: %v", err)
		}
		defer sub.Cancel()


		// Goroutine for handling incoming messages from the subscription topic
		go func() {
		for {
			msg, err := sub.Next(ctx)
			if err != nil {
				log.Printf("Failed to read next message from subscription topic: %v", err)
				continue
			}
			            // Ignore messages from itself
            		if msg.ReceivedFrom == h.ID() {
                		continue
            		}

			// Print the received message along with the peerID
			fmt.Printf("%s: %s\n", msg.ReceivedFrom, string(msg.Data))
		}
		}()

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

	        bool_flag, truncatedHash, err = verifyArgon2HashRemotely(hashToVerify, key)
		if bool_flag {
			//var shaHash = sha256.Sum256([]byte(hashToVerify))
			// Convert the SHA256 hash to a hexadecimal string and take the first 5 characters
			// truncatedHash = hex.EncodeToString(shaHash[:])[:5]
			fmt.Printf("Verification success for block ID %s\n", blockID)
		} else {
			fmt.Printf("Local Verification failed for block ID %s\n", blockID)
			continue
		}
		// Stop the performance timer and print the elapsed time
		elapsed := time.Since(start)
		elapsedInMilliseconds := elapsed / time.Millisecond
		timeStr := strconv.FormatInt(int64(elapsedInMilliseconds), 10)

		fmt.Printf("Local Verification time for block ID %s: %s %s\n", blockID,truncatedHash,elapsed)

		if bool_flag {

			for _, peerID := range kadDHT.Host().Peerstore().Peers() {
                        	latency := kadDHT.Host().Peerstore().LatencyEWMA(peerID)
				if latency > 0 {
                        	    fmt.Printf("Latency to peer %s: %s %v\n", peerID.String(), latency, h.Peerstore().Addrs(peerID)[0])
					}
				}

			responseData := map[string]interface{}{
				"peer_id":    h.ID().String(),
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
			
			// Publish the JSON response to the messages topic
			if err := subscriptionTopic.Publish(ctx, jsonData); err != nil {
				log.Printf("Failed to publish to messages topic: %v", err)
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

