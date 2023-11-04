package main

import (
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	log0 "github.com/ipfs/go-log/v2"
	"github.com/joho/godotenv"
	"github.com/libp2p/go-libp2p"
	dht "github.com/libp2p/go-libp2p-kad-dht"
	pubsub "github.com/libp2p/go-libp2p-pubsub"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/peerstore"
	drouting "github.com/libp2p/go-libp2p/p2p/discovery/routing"
	"github.com/multiformats/go-multiaddr"
	"github.com/redis/go-redis/v9"
	"github.com/samber/lo"
	"log"
	"os"
	"strings"
	"sync"
)

func loadPeerParams(path string, logger log0.EventLogger) (multiaddr.Multiaddr, crypto.PrivKey, string) {
	content, err := os.ReadFile(path + "/peer.json")
	if err != nil {
		logger.Error("Error when opening file: ", err)
	}
	err = godotenv.Load(path + "/.env")
	if err != nil {
		logger.Error("Error loading ENV: ", err)
	}
	port := os.Getenv("PORT")
	if port == "" {
		port = "10330"
	}

	// Now let's unmarshall the data into `peerId`
	var peerId PeerId
	err = json.Unmarshal(content, &peerId)
	if err != nil {
		logger.Error("Error reading Peer config file: ", err)
	}
	logger.Info("PeerId: ", peerId.Id)

	addr, err := multiaddr.NewMultiaddr(fmt.Sprintf("/ip4/0.0.0.0/udp/%s/quic-v1", port))
	if err != nil {
		logger.Error("Error making address: ", err)
	}

	sk, err := base64.StdEncoding.DecodeString(peerId.PrivKey)
	if err != nil {
		logger.Error("Error base64-decoding pk: ", err)
	}

	privKey, err := crypto.UnmarshalPrivateKey(sk)
	if err != nil {
		logger.Error("Error converting pk: ", err)
	}
	if err != nil {
		panic(err)
	}

	return addr, privKey, peerId.Id
}

func prepareBootstrapAddresses(path string, logger log0.EventLogger) []string {
	err := godotenv.Load(path + "/.env")
	if err != nil {
		logger.Error("Error loading ENV: ", err)
	}
	notEmpty := func(item string, index int) bool {
		return item != ""
	}
	bootstrapHosts := lo.Filter[string](strings.Split(os.Getenv("BOOTSTRAP_HOSTS"), ","), notEmpty)
	bootstrapPorts := lo.Filter[string](strings.Split(os.Getenv("BOOTSTRAP_PORTS"), ","), notEmpty)
	bootstrapPeers := lo.Filter[string](strings.Split(os.Getenv("BOOTSTRAP_PEERS"), ","), notEmpty)

	var destinations []string
	for i, peerId := range bootstrapPeers {
		destinations = append(
			destinations,
			fmt.Sprintf(
				"/ip4/%s/tcp/%s/p2p/%s",
				bootstrapHosts[i],
				bootstrapPorts[i],
				peerId,
			),
			fmt.Sprintf(
				"/ip4/%s/tcp/%s/ws/p2p/%s",
				bootstrapHosts[i],
				bootstrapPorts[i],
				peerId,
			),
			fmt.Sprintf(
				"/ip4/%s/udp/%s/quic-v1/p2p/%s",
				bootstrapHosts[i],
				bootstrapPorts[i],
				peerId,
			),
		)
	}
	return destinations
}

var toAddrInfo = func(destination string, _ int) peer.AddrInfo {
	address, err := multiaddr.NewMultiaddr(destination)
	if err != nil {
		log.Println(err)
	}
	info, err := peer.AddrInfoFromP2pAddr(address)
	if err != nil {
		log.Println(err)
	}
	return *info
}

var toAddrInfoPtr = func(destination string, _ int) *peer.AddrInfo {
	address, err := multiaddr.NewMultiaddr(destination)
	if err != nil {
		log.Println(err)
	}
	info, err := peer.AddrInfoFromP2pAddr(address)
	if err != nil {
		log.Println(err)
	}
	return info
}

func connectToPeer(ctx context.Context, destination string) {
	h := ctx.Value("host").(host.Host)
	logger := ctx.Value("logger").(log0.EventLogger)

	// Turn the destination into a multiaddr.
	info := toAddrInfo(destination, 0)
	// Add the destination's peer multiaddress in the peerstore.
	// This will be used during connection and stream creation by Libp2p.
	h.Peerstore().AddAddrs(info.ID, info.Addrs, peerstore.PermanentAddrTTL)
	err := h.Connect(ctx, info)
	if err != nil {
		logger.Warn("Error connecting: ", err)
	}
}

func setupConnections(ctx context.Context, destinations []string) {
	logger := ctx.Value("logger").(log0.EventLogger)

	logger.Info("destinations", destinations)

	for _, dest := range destinations {
		connectToPeer(ctx, dest)
		logger.Info("Connect to: ", dest)
	}
}

func setupDB(path string, ro bool, logger log0.EventLogger) *sql.DB {
	err := godotenv.Load(path + "/.env")
	var dbPath = ""
	if err != nil {
		err = nil
	}
	dbPath = os.Getenv("DB_LOCATION")
	if dbPath == "" {
		dbPath = "file:" + path + "/blockchain.db?cache=shared&"
	} else {
		dbPath = "file:" + dbPath + "?cache=shared&"
	}
	if ro {
		// add read-only flag
		dbPath += "mode=ro"
	} else {
		dbPath += "_journal_mode=WAL&mode=rwc"
	}
	logger.Info("DB path: ", dbPath)
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		log.Fatal("Error when opening DB file: ", err)
	}

	if !ro {
		_, err = db.Exec(createBlockchainTableSql)
		if err != nil {
			log.Fatal("Error when checking/creating table: ", err)
		}
	}

	maxHeight := getCurrentHeight(db)
	logger.Info("HGHT ", maxHeight)

	return db
}

func setupHashesDB(path string, ro bool, logger log0.EventLogger, state *NetworkState) *sql.DB {
	err := godotenv.Load(path + "/.env")
	var dbPath = ""
	if err != nil {
		err = nil
	}
	dbPath = os.Getenv("DBH_LOCATION")
	if dbPath == "" {
		dbPath = "file:" + path + "/blocks.db?cache=shared&"
	} else {
		dbPath = "file:" + dbPath + "?cache=shared&"
	}
	if ro {
		// add read-only flag
		dbPath += "mode=ro"
	} else {
		dbPath += "_journal_mode=WAL&mode=rwc"
	}

	logger.Info("DBH path: ", dbPath)
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		log.Fatal("Error when opening hashes DB file: ", err)
	}

	if !ro {
		_, err = db.Exec(createHashesTableSql)
		if err != nil {
			log.Fatal("Error creating hashes table: ", err)
		}

		_, err = db.Exec(createXunisTableSql)
		if err != nil {
			log.Fatal("Error creating xunis table: ", err)
		}
	}

	state.LastHashId = uint64(getLatestHashId(db))
	state.LastXuniId = uint64(getLatestXuniId(db))
	logger.Info("LAST ", state)

	return db
}

func setupControlDB(path string, logger log0.EventLogger) *sql.DB {
	err := godotenv.Load(path + "/.env")
	var dbPath = ""
	if err != nil {
		err = nil
	}
	dbPath = os.Getenv("CONTROLDB_LOCATION")
	if dbPath == "" {
		dbPath = path + "/control.db"
	}

	logger.Info("CONTROL DB path: ", dbPath)
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		log.Fatal("Error when opening control DB file: ", err)
	}
	_, err = db.Exec(createControlTableSql)
	if err != nil {
		log.Fatal("Error creating control table: ", err)
	}

	// TODO: remove after tests
	/*
		testRange := RangeRecord{Id: 1, BlocksRange: "0-1", Hash: "hash", Difficulty: 1, Node: "myself"}
		err0 := insertRangeRecord(db, testRange)
		if err0 != nil {
			logger.Warn("err ", err0)
		}
	*/

	// _ = db.Close()
	// db, err = sql.Open("sqlite3", dbPath+"&mode=ro")

	return db
}

func setupHost(privKey crypto.PrivKey, addr multiaddr.Multiaddr) host.Host {
	h, err := libp2p.New(
		libp2p.ListenAddrs(addr),
		libp2p.Identity(privKey),
	)
	if err != nil {
		log.Fatal("Error starting Peer: ", err)
	}
	return h
}

func setupRedis(ctx context.Context) (rdb *redis.Client) {
	logger := ctx.Value("logger").(log0.EventLogger)
	rdb = redis.NewClient(&redis.Options{
		Addr:     "localhost:6379",
		Password: "", // no password set
		DB:       0,  // use default DB
	})
	err := rdb.Set(ctx, "test", "1", 0).Err()
	if err != nil {
		logger.Warn("Error talking to Redis, probably not present: ", err)
		return nil
	}

	val, err := rdb.Get(ctx, "test").Result()
	if err != nil {
		panic(err)
	}
	if val != "1" {
		panic("RDS assert error")
	}
	return rdb
}

func setupDiscovery(ctx context.Context, destinations []string) *drouting.RoutingDiscovery {
	h := ctx.Value("host").(host.Host)
	dhTable := ctx.Value("dht").(*dht.IpfsDHT)
	logger := ctx.Value("logger").(log0.EventLogger)

	// Let's connect to the bootstrap nodes first. They will tell us about the
	// other nodes in the network.

	var wg sync.WaitGroup
	for i, peerAddr := range destinations {
		peerInfo := toAddrInfo(peerAddr, i)
		wg.Add(1)
		go func() {
			defer wg.Done()
			h.Peerstore().AddAddrs(peerInfo.ID, peerInfo.Addrs, peerstore.PermanentAddrTTL)
			if err := h.Connect(ctx, peerInfo); err != nil {
				logger.Warn(err)
			} else {
				logger.Info("Connection established with bootstrap node:", peerInfo)
				/*
					r, err := dhTable.RoutingTable().TryAddPeer(peerInfo.ID, true, false)
					if err != nil {
						logger.Warn(err)
					} else if r {
						logger.Info("Added to RT: ", peerInfo.ID)
					}
				*/
			}
		}()
	}
	wg.Wait()

	// We use a rendezvous point "meet me here" to announce our location.
	// This is like telling your friends to meet you at the Eiffel Tower.
	routingDiscovery := drouting.NewRoutingDiscovery(dhTable)
	t, _ := routingDiscovery.Advertise(ctx, rendezvousString)
	logger.Infof("DHT started announcing for %d", t)

	/*
		logger.Info("Searching for other peers")
		peerChan, err := routingDiscovery.FindPeers(ctx, rendezvousString)
		if err != nil {
			logger.Warn(err)
		}

		for p := range peerChan {
			logger.Info("Peer candidate: ", p)
			if p.ID == h.ID() || hasDestination(destinations, p.ID.String()) {
				continue
			}
			logger.Info("Found peer:", p)
			h.Peerstore().AddAddrs(p.ID, p.Addrs, peerstore.PermanentAddrTTL)
			err = h.Connect(ctx, p)
			if err != nil {
				logger.Warn("Error connecting to peer: ", err)
			}
			logger.Info("Connected to:", p)
		}
	*/

	return routingDiscovery
}

func subscribeToTopics(ps *pubsub.PubSub, logger log0.EventLogger) (topics Topics, subs Subs) {
	var err error
	topics.blockHeight, err = ps.Join("block_height")
	if err != nil {
		logger.Error("Error joining topic 'block_height'", err)
	}
	subs.blockHeight, err = topics.blockHeight.Subscribe()
	if err != nil {
		logger.Error("Error subscribing to topic 'block_height'", err)
	}

	topics.get, err = ps.Join("get")
	if err != nil {
		logger.Error("Error joining topic 'get'", err)
	}
	subs.get, err = topics.get.Subscribe()
	if err != nil {
		logger.Error("Error subscribing to topic 'get'", err)
	}

	topics.data, err = ps.Join("data")
	if err != nil {
		logger.Error("Error joining topic 'data'", err)
	}
	subs.data, err = topics.data.Subscribe()
	if err != nil {
		logger.Error("Error subscribing to topic 'data'", err)
	}

	topics.newHash, err = ps.Join("new_hash")
	if err != nil {
		logger.Error("Error joining topic 'new_hash'", err)
	}
	subs.newHash, err = topics.newHash.Subscribe()
	if err != nil {
		logger.Error("Error subscribing to topic 'new_hash'", err)
	}

	topics.newXuni, err = ps.Join("new_xuni")
	if err != nil {
		logger.Error("Error joining topic 'new_xuni'", err)
	}
	subs.newXuni, err = topics.newXuni.Subscribe()
	if err != nil {
		logger.Error("Error subscribing to topic 'new_xuni'", err)
	}

	topics.shift, err = ps.Join("shift")
	if err != nil {
		logger.Error("Error joining topic 'shift'", err)
	}
	subs.shift, err = topics.shift.Subscribe()
	if err != nil {
		logger.Error("Error subscribing to topic 'shift'", err)
	}

	topics.getRaw, err = ps.Join("get_raw")
	if err != nil {
		logger.Error("Error joining topic 'get_raw'", err)
	}
	subs.getRaw, err = topics.getRaw.Subscribe()
	if err != nil {
		logger.Error("Error subscribing to topic 'get_raw'", err)
	}

	topics.control, err = ps.Join("control")
	if err != nil {
		logger.Error("Error joining topic 'control'", err)
	}
	subs.control, err = topics.control.Subscribe()
	if err != nil {
		logger.Error("Error subscribing to topic 'control'", err)
	}

	if err != nil {
		panic(err)
	}
	return
}
