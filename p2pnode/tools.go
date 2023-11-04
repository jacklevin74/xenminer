package main

import (
	"bufio"
	"context"
	"crypto/rand"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	log0 "github.com/ipfs/go-log/v2"
	"github.com/joho/godotenv"
	"github.com/libp2p/go-libp2p/core/crypto"
	"github.com/libp2p/go-libp2p/core/host"
	"github.com/libp2p/go-libp2p/core/network"
	"github.com/libp2p/go-libp2p/core/peer"
	"github.com/libp2p/go-libp2p/core/protocol"
	"log"
	"os"
	"strconv"
	"time"
)

func checkDir(path0 string) bool {
	var path string
	if path0 == "" {
		path = ".node"
	} else {
		path = path0
	}
	// check if dir doesn't exist; if no, create it
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		return false
	} else {
		return true
	}
}

func initDir(path0 string, logger log0.EventLogger) string {
	logger.Info("Initializing node cfg and DB")
	var path string
	if path0 == "" {
		path = ".node"
	} else {
		path = path0
	}
	// check if dir doesn't exist; if no, create it
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		logger.Info("Created Node dir")
		err := os.Mkdir(path, os.ModePerm)
		if err != nil {
			logger.Warn(err)
		}
	} else {
		logger.Info("Node dir exists, skipping")
	}
	return path
}

func checkOrCreateDb(path string, name string, logger log0.EventLogger) {
	pathToDb := path + "/" + name
	if _, err := os.Stat(pathToDb); errors.Is(err, os.ErrNotExist) {
		db, err := sql.Open("sqlite3", pathToDb)
		if err != nil {
			logger.Fatalf("Error when opening DB file %s: %s", name, err)
		}
		_, err = db.Exec(initDbSql)
		if err != nil {
			logger.Fatalf("Error when init DB file %s: %s", name, err)
		}
		err = db.Close()
		if err != nil {
			logger.Fatalf("Error closing DB %s: %s", name, err)
		}
		logger.Infof("Created DB %s", name)
	} else {
		logger.Infof("DB %s exists", name)
	}
}

func checkOrCreatePeerId(path string, logger log0.EventLogger) {
	pathToPeerId := path + "/peer.json"

	if _, err := os.Stat(pathToPeerId); errors.Is(err, os.ErrNotExist) {
		priv, pub, err := crypto.GenerateEd25519Key(rand.Reader)
		if err != nil {
			log.Fatal("Error when generating keypair: ", err)
		}
		privBytes, err := crypto.MarshalPrivateKey(priv)
		pubBytes, err := crypto.MarshalPublicKey(pub)
		id, err := peer.IDFromPublicKey(pub)
		if err != nil {
			log.Fatal("Error when converting keypair: ", err)
		}
		privKey := base64.StdEncoding.EncodeToString(privBytes)
		pubKey := base64.StdEncoding.EncodeToString(pubBytes)
		peerId := PeerId{Id: id.String(), PrivKey: privKey, PubKey: pubKey}
		bytes, err := json.Marshal(&peerId)
		if err != nil {
			log.Fatal("Error when converting peerId: ", err)
		}
		err = os.WriteFile(pathToPeerId, bytes, os.ModePerm)
		if err != nil {
			log.Fatal("Error writing peerId file: ", err)
		}
		logger.Info("Created peerId file")
	} else {
		logger.Info("peerId file exists, skipping")
	}
}

func checkOrCreateEnvFile(path string, logger log0.EventLogger) {
	pathToEnv := path + "/.env"

	if _, err := os.Stat(pathToEnv); errors.Is(err, os.ErrNotExist) {
		bytes := []byte(
//			"BOOTSTRAP_HOSTS=35.87.16.125,34.208.84.148,209.124.84.6\n" +
//				"BOOTSTRAP_PORTS=10330,10330,10330\n" +
//				"BOOTSTRAP_PEERS=12D3KooWEmj8Qy3G68gKTHroiMEn59HiziqEN7QdiHMkviBEDr69,12D3KooWGymdz8QqwdN9GuyzLK7sxcXAtBNb4RndhNGUk3Vbt1Nu,12D3KooWLGpxvuNUmMLrQNKTqvxXbXkR1GceyRSpQXd8ZGmprvjH")
			"BOOTSTRAP_HOSTS=209.124.84.6\n" +
				"BOOTSTRAP_PORTS=10330\n" +
				"BOOTSTRAP_PEERS=12D3KooWLGpxvuNUmMLrQNKTqvxXbXkR1GceyRSpQXd8ZGmprvjH")
		err := os.WriteFile(pathToEnv, bytes, 0777)
		if err != nil {
			log.Fatal("Error writing ENV file: ", err)
		}
		logger.Info("Created ENV file")
	} else {
		logger.Info("ENV file exists, skipping")
	}
}

func initNode(path0 string, logger log0.EventLogger) {
	path := initDir(path0, logger)

	checkOrCreateDb(path, "blockchain.db", logger)
	checkOrCreateDb(path, "blocks.db", logger)
	checkOrCreateDb(path, "control.db", logger)

	checkOrCreatePeerId(path, logger)
	checkOrCreateEnvFile(path, logger)
}

func resetBlockchainDb(path0 string, logger log0.EventLogger) {
	err := godotenv.Load(path0 + "/.env")
	var dbPath = ""
	if err != nil {
		err = nil
	}
	dbPath = os.Getenv("DB_LOCATION")
	if dbPath == "" {
		dbPath = path0 + "/blockchain.db"
	}

	logger.Info("Resetting node to block height 0: ", dbPath)
	db, err := sql.Open("sqlite3", dbPath)
	if err != nil {
		log.Fatal("Error when opening DB file: ", err)
	}
	_, err = db.Exec(resetDbSql)
	if err != nil {
		log.Fatal("Error when resetting DB: ", err)
	}
	err = db.Close()
	if err != nil {
		log.Fatal("Error closing DB: ", err)
	}
	logger.Info("Reset DB")
}

func resetHashesDb(path0 string, logger log0.EventLogger) {
	err := godotenv.Load(path0 + "/.env")
	var dbhPath = ""
	if err != nil {
		err = nil
	}
	dbhPath = os.Getenv("DBH_LOCATION")
	if dbhPath == "" {
		dbhPath = "file:" + path0 + "/blocks.db?cache=shared&"
	} else {
		dbhPath = "file:" + dbhPath + "?cache=shared&"
	}

	logger.Info("DBH path: ", dbhPath)
	dbh, err := sql.Open("sqlite3", dbhPath)
	if err != nil {
		log.Fatal("Error when opening hashes DB file: ", err)
	}
	_, err = dbh.Exec(resetHashesSql)
	if err != nil {
		log.Fatal("Error resetting hashes DB: ", err)
	}
	logger.Info("Hashes DB was reset")
	_, err = dbh.Exec(resetXunisSql)
	if err != nil {
		log.Fatal("Error resetting xunis DB: ", err)
	}
	logger.Info("Xunis DB was reset")
	err = dbh.Close()
	if err != nil {
		log.Fatal("Error closing DBH: ", err)
	} else {
		logger.Info("DBH was closed")
	}
}

func syncHashes(path0 string, logger log0.EventLogger) {
	err := godotenv.Load(path0 + "/.env")
	var dbPath = ""
	var dbhPath = ""
	if err != nil {
		err = nil
	}
	dbPath = os.Getenv("DB_LOCATION")
	if dbPath == "" {
		dbPath = path0 + "/blockchain.db?cache=shared&mode=ro"
	} else {
		dbhPath = "file:" + dbhPath + "?cache=shared&mode=ro"
	}

	logger.Info("Opening DB: ", dbPath)
	db, err := sql.Open("sqlite3", "file:"+dbPath)
	if err != nil {
		log.Fatal("Error when opening DB file: ", err)
	}

	dbhPath = os.Getenv("DBH_LOCATION")
	if dbhPath == "" {
		dbhPath = "file:" + path0 + "/blocks.db?cache=shared&mode=rwc&_journal_mode=WAL"
	} else {
		dbhPath = "file:" + dbhPath + "?cache=shared&mode=rwc&_journal_mode=WAL"
	}

	logger.Info("DBH path: ", dbhPath)
	dbh, err := sql.Open("sqlite3", dbhPath)
	if err != nil {
		log.Fatal("Error when opening hashes DBH file: ", err)
	}

	_, err = dbh.Exec(createHashesTableSql)
	if err != nil {
		log.Fatal("Error creating hashes table: ", err)
	}

	_, err = dbh.Exec(createXunisTableSql)
	if err != nil {
		log.Fatal("Error creating xunis table: ", err)
	}

	logger.Info("Copying hashes...")
	c := getAllBlocks(db)
	count := 0
	xen11 := 0
	xuni := 0
	for row := range c {
		var records []Record
		err := json.Unmarshal([]byte(row.RecordsJson), &records)
		for _, rec := range records {
			// datetime, err := time.Parse(time.RFC3339, strings.Replace(rec.Date, " ", "T", 1)+"Z")
			// if err != nil {
			// 	log.Println("Error parsing time: ", err)
			// }
			var hashRec HashRecord
			if rec.XuniId != nil {
				hashRec = HashRecord{
					Id: uint(int64(*rec.XuniId)),
					// CreatedAt:    uint(datetime.Unix()),
					CreatedAt:    rec.Date,
					Account:      rec.Account,
					HashToVerify: rec.HashToVerify,
					Key:          rec.Key,
				}
				err = insertXuniRecord(dbh, hashRec)
				xuni++
			} else {
				hashRec = HashRecord{
					Id: uint(int64(*rec.BlockId)),
					// CreatedAt:    uint(datetime.Unix()),
					CreatedAt:    rec.Date,
					Account:      rec.Account,
					HashToVerify: rec.HashToVerify,
					Key:          rec.Key,
				}
				err = insertHashRecord(dbh, hashRec)
				xen11++
			}
			if err != nil {
				log.Println("\nError inserting: ", err)
			} else {
				count++
				fmt.Printf("\rProcessing rec (%d/%d): %d", xen11, xuni, count)
			}
		}
		if err != nil {
			log.Println("\nError converting: ", err)
		}
	}
	fmt.Println()

	err = db.Close()
	if err != nil {
		log.Fatal("Error closing DB: ", err)
	}
	logger.Info("Done with DB")
	err = dbh.Close()
	if err != nil {
		log.Fatal("Error closing DBH: ", err)
	}
	logger.Info("Done with DBH")
}

func doSend(ctx context.Context, id peer.ID) {
	h := ctx.Value("host").(host.Host)
	logger := ctx.Value("logger").(log0.EventLogger)

	c := make(chan []byte)
	buf := make([]byte, 512)
	t := time.NewTicker(1 * time.Second)
	count := 0
	// series := 0
	// times := map[int]int{}
	// mutex := sync.RWMutex{}
	start := 0
	delta := 0

	conn, err := h.NewStream(context.Background(), id, protocol.TestingID)
	if err != nil {
		logger.Warn("Err in conn ", err)
	}

	rw := bufio.NewReadWriter(bufio.NewReader(conn), bufio.NewWriter(conn))

	logger.Info("Connection ", conn.Stat())

	if err != nil {
		logger.Fatal("Error: ", err)
	}

	go func() {
		for {
			_, err := rand.Read(buf)
			if err != nil {
				logger.Warn("Err in rand ", err)
			}
			c <- buf
		}
	}()

	go func() {
		for {
			s, err := rw.ReadString('\n')
			if err != nil {
				logger.Warn("Err in rand ", err)
			}
			_, err = strconv.Atoi(s)
			if start != 0 {
				delta = int(time.Now().UnixMilli()) - start
				start = 0
			}
			// mutex.RLock()
			// logger.Infof("Read: %d, delta: %d", i, int(time.Now().UnixMilli())-times[i])
			// mutex.RUnlock()
		}
	}()

	for {
		select {
		case <-t.C:
			logger.Infof("%d bytes/s, delta: %d", count, delta)
			count = 0

		case bytes := <-c:
			n, err := rw.Write(bytes)
			count += n
			// mutex.Lock()
			// times[series] = int(time.Now().UnixMilli())
			// mutex.Unlock()
			// series += 1
			if start == 0 {
				start = int(time.Now().UnixMilli())
			}
			if err != nil {
				logger.Warn("Error: ", err)
				return
			}

		case <-ctx.Done():
			logger.Info("DONE")
		}
	}
}

func decode(rw *bufio.ReadWriter, logger log0.EventLogger) error {
	buff := make([]byte, 512)
	t := time.NewTicker(1 * time.Second)
	defer t.Stop()
	quit := make(chan struct{})
	count := 0
	series := 0

	go func() {
		for {
			n, err := rw.Read(buff)
			if err != nil {
				logger.Warn("read err: ", err)
				quit <- struct{}{}
				return
			} else {
				count += n
				_, err = rw.WriteString(fmt.Sprintf("%d\n", series))
				series++
				if err != nil {
					logger.Warn("write err: ", err)
				}
			}
		}
	}()

	for {
		select {
		case <-t.C:
			logger.Infof("Rec %d byte/s", count)
			count = 0
		case <-quit:
			t.Stop()
			return errors.New("stopped")
		}
	}

}

func doReceive(ctx context.Context) {
	h := ctx.Value("host").(host.Host)
	logger := ctx.Value("logger").(log0.EventLogger)

	h.SetStreamHandler(protocol.TestingID, func(s network.Stream) {
		logger.Info("listener received new stream", s.Stat())
		rw := bufio.NewReadWriter(bufio.NewReader(s), bufio.NewWriter(s))
		log.Println("Reading stream")
		err := decode(rw, logger)
		logger.Info("Stream: ", err)
	})
	logger.Info("Listening")

	<-ctx.Done()
}
