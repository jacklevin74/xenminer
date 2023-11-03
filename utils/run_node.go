package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"bufio"
	"time"
	"log"
	"strings"

	"github.com/libp2p/go-libp2p"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p-pubsub"
	//"github.com/libp2p/go-libp2p/core/host"
	"github.com/multiformats/go-multiaddr"
	//"github.com/libp2p/go-libp2p-kbucket"

	dht "github.com/libp2p/go-libp2p-kad-dht"
	//log2 "github.com/ipfs/go-log/v2"
)




func handleStream(s network.Stream) {
	buf := bufio.NewReader(s)
	for {
		str, err := buf.ReadString('\n')
		if err != nil {
			log.Printf("Error reading from buffer: %s %s", err)
			break
		}

		addr, err := multiaddr.NewMultiaddr(strings.TrimSpace(str))
		if err == nil {
			info, err := peer.AddrInfoFromP2pAddr(addr)
			if err == nil {
				log.Printf("Received peer address: %s", info)
			}
		}
	}
}

func main() {
	//log2.SetAllLoggers(log2.LevelDebug)
	ctx := context.Background()

	// Accept the bootstrap node's multiaddress from the command line.
	bootstrapAddr := flag.String("bootstrap", "", "The multiaddress of the bootstrap node")
	flag.Parse()

	if *bootstrapAddr == "" {
		fmt.Println("Bootstrap multiaddress must be provided with -bootstrap flag")
		os.Exit(1)
	}

	// Create a new libp2p Host with default options
	h, err := libp2p.New() // Notice we removed the context from here
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create host: %v\n", h,err)
		os.Exit(1)
	}
	h.SetStreamHandler("/ipfs/kad/1.0.0", handleStream)


	ps, err := pubsub.NewGossipSub(ctx, h)
	if err != nil {
		log.Fatalf("Failed to create pubsub: %v", err)
	}

	topic, err := ps.Join("timestamps")
	if err != nil {
		log.Fatalf("Failed to join topic: %v", err)
	}

	subscription, err := topic.Subscribe()
	if err != nil {
		log.Fatalf("Failed to subscribe to topic: %v", err)
	}

	// You might want to handle incoming messages in a separate goroutine
	go func() {
		for {
			msg, err := subscription.Next(ctx)
			if err != nil {
				log.Printf("Failed to get next message: %v", err)
				continue
			}
			// Process the message, for example, print the timestamp
			log.Printf("Received timestamp: %s from peer: %s", string(msg.Data), msg.GetFrom())
		}
	}()



	defer h.Close()

	// Connect to the bootstrap node
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

	// Create a new DHT instance for the peer, in client mode
	var dhtOptions []dht.Option
	dhtOptions = append(dhtOptions, dht.Mode(dht.ModeServer))
	dhtOptions = append(dhtOptions, dht.BootstrapPeers(*bootstrapAI))

	kademliaDHT, err := dht.New(ctx, h, dhtOptions...)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Failed to create the DHT: %v\n", err)
		os.Exit(1)
	}

	// Bootstrap the DHT
	if err := kademliaDHT.Bootstrap(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Failed to bootstrap DHT: %v\n", err)
		os.Exit(1)
	}


	// This would replace your for loop where you send timestamps directly
	for {
		timestamp := time.Now().Format(time.RFC3339)
		// Publish the timestamp to all peers subscribed to the "timestamps" topic
		err := topic.Publish(ctx, []byte(timestamp))
		if err != nil {
			log.Printf("Failed to publish timestamp: %v", err)
		} else {
			log.Printf("Published timestamp: %s", timestamp)
		}
		time.Sleep(3 * time.Second)
	}


	// Run indefinitely
	select {}
}
