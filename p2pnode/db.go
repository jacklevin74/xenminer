package main

import (
	"database/sql"
	"fmt"
	"log"
)

func getBlock(db *sql.DB, blockId uint) (*Block, error) {
	row := db.QueryRow(getRowBlockchainSql, fmt.Sprintf("%d", blockId))
	var block Block
	err := row.Scan(&block.Id, &block.Timestamp, &block.PrevHash, &block.MerkleRoot, &block.RecordsJson, &block.BlockHash)
	if err != nil {
		return nil, err
	}
	return &block, nil
}

func getPrevBlock(db *sql.DB, block *Block) (*Block, error) {
	prevRow := db.QueryRow(getRowBlockchainSql, fmt.Sprintf("%d", block.Id-1))
	var prevBlock Block
	err := prevRow.Scan(
		&prevBlock.Id,
		&prevBlock.Timestamp,
		&prevBlock.PrevHash,
		&prevBlock.MerkleRoot,
		&prevBlock.RecordsJson,
		&prevBlock.BlockHash,
	)
	if err != nil {
		return nil, err
	}
	return &prevBlock, nil
}

func insertBlock(db *sql.DB, block *Block) error {
	_, err := db.Exec(
		insertBlockchainSql,
		block.Id,
		block.Timestamp,
		block.PrevHash,
		block.MerkleRoot,
		block.RecordsJson,
		block.BlockHash,
	)
	return err
}

func getMissingBlockIds(db *sql.DB) []uint {
	currentHeight := getCurrentHeight(db)
	rows, err := db.Query(getMissingRowIdsBlockchainSql)
	if err != nil {
		log.Println("Error when querying DB: ", err)
		return make([]uint, 0)
	}
	var blockId uint
	defer func(rows *sql.Rows) {
		err := rows.Close()
		if err != nil {
			log.Println("Error when closing rows: ", err)
		}
	}(rows)
	var blocks []uint
	for rows.Next() {
		err = rows.Scan(&blockId)
		if blockId < currentHeight {
			// avoid repeatedly asking for next block if the DB is synced
			blocks = append(blocks, blockId)
		}
	}
	return blocks
}

func getMissingHashIds(db *sql.DB) []uint {
	currentLastId := getLatestHashId(db)
	rows, err := db.Query(getMissingHashRowIdsSql)
	if err != nil {
		log.Println("Error when querying DB: ", err)
		return make([]uint, 0)
	}
	var hashId uint
	defer func(rows *sql.Rows) {
		err := rows.Close()
		if err != nil {
			log.Println("Error when closing rows: ", err)
		}
	}(rows)
	var ids []uint
	for rows.Next() {
		err = rows.Scan(&hashId)
		if hashId < currentLastId {
			// avoid repeatedly asking for next block if the DB is synced
			ids = append(ids, hashId)
		}
	}
	return ids
}

func getMissingXuniIds(db *sql.DB) []uint {
	currentLastId := getLatestXuniId(db)
	rows, err := db.Query(getMissingXuniRowIdsSql)
	if err != nil {
		log.Println("Error when querying DB: ", err)
		return make([]uint, 0)
	}
	var hashId uint
	defer func(rows *sql.Rows) {
		err := rows.Close()
		if err != nil {
			log.Println("Error when closing rows: ", err)
		}
	}(rows)
	var ids []uint
	for rows.Next() {
		err = rows.Scan(&hashId)
		if hashId < currentLastId {
			// avoid repeatedly asking for next block if the DB is synced
			ids = append(ids, hashId)
		}
	}
	return ids
}

func getCurrentHeight(db *sql.DB) uint {
	rows, err := db.Query(getMaxHeightBlockchainSql)
	if err != nil {
		log.Println("Error when querying DB: ", err)
		return 0
	}
	var height Height
	defer func(rows *sql.Rows) {
		err := rows.Close()
		if err != nil {
			log.Println("Error when closing rows: ", err)
		}
	}(rows)
	rows.Next()
	err = rows.Scan(&height.Max)
	if err != nil {
		log.Println("Error retrieving data from DB: ", err)
	}
	if height.Max.Valid {
		return uint(height.Max.Int32)
	} else {
		return 0
	}
}

func getLatestHashId(db *sql.DB) uint {
	rows, err := db.Query(getLatestHashIdSql)
	if err != nil {
		log.Println("Error when querying HDB: ", err)
		return 0
	}
	var height Height
	defer func(rows *sql.Rows) {
		err := rows.Close()
		if err != nil {
			log.Println("Error when closing rows: ", err)
		}
	}(rows)
	rows.Next()
	err = rows.Scan(&height.Max)
	if err != nil {
		log.Println("Error retrieving data from HDB: ", err)
	}
	if height.Max.Valid {
		return uint(height.Max.Int32)
	} else {
		return 0
	}
}

func getLatestXuniId(db *sql.DB) uint {
	rows, err := db.Query(getLatestXuniIdSql)
	if err != nil {
		log.Println("Error when querying HDB: ", err)
		return 0
	}
	var height Height
	defer func(rows *sql.Rows) {
		err := rows.Close()
		if err != nil {
			log.Println("Error when closing rows: ", err)
		}
	}(rows)
	rows.Next()
	err = rows.Scan(&height.Max)
	if err != nil {
		log.Println("Error retrieving data from HDB: ", err)
	}
	if height.Max.Valid {
		return uint(height.Max.Int32)
	} else {
		return 0
	}
}

func getLatestHash(db *sql.DB) *HashRecord {
	rows, err := db.Query(getLatestHashSql)
	if err != nil {
		log.Println("Error when querying HDB: ", err)
		return nil
	}
	var hash HashRecord
	defer func(rows *sql.Rows) {
		err := rows.Close()
		if err != nil {
			log.Println("Error when closing rows: ", err)
		}
	}(rows)
	rows.Next()
	err = rows.Scan(
		&hash.Id,
		&hash.HashToVerify,
		&hash.Key,
		&hash.Account,
		&hash.CreatedAt,
	)
	if err != nil {
		log.Println("Error retrieving data from HDB: ", err)
	}
	return &hash
}

func getLatestRange(db *sql.DB) *RangeRecord {
	rows, err := db.Query(getLatestRangeSql)
	if err != nil {
		log.Println("Error when querying Control DB: ", err)
		return nil
	}
	var record RangeRecord
	defer func(rows *sql.Rows) {
		err := rows.Close()
		if err != nil {
			log.Println("Error when closing rows: ", err)
		}
	}(rows)
	rows.Next()
	err = rows.Scan(
		&record.Id,
		&record.Node,
		&record.BlocksRange,
		&record.Hash,
		&record.Difficulty,
		&record.Ts,
	)
	if err != nil {
		log.Println("Error retrieving data from Control DB: ", err)
		return nil
	}
	return &record
}

func insertRangeRecord(db *sql.DB, record RangeRecord) error {
	_, err := db.Exec(
		insertRangeSql,
		// record.Id,
		record.Node,
		record.BlocksRange,
		record.Hash,
		record.Difficulty,
		record.Ts,
	)
	return err
}

func getLatestXuni(db *sql.DB) *HashRecord {
	rows, err := db.Query(getLatestXuniSql)
	if err != nil {
		log.Println("Error when querying HDB: ", err)
		return nil
	}
	var hash HashRecord
	defer func(rows *sql.Rows) {
		err := rows.Close()
		if err != nil {
			log.Println("Error when closing rows: ", err)
		}
	}(rows)
	rows.Next()
	err = rows.Scan(
		&hash.Id,
		&hash.HashToVerify,
		&hash.Key,
		&hash.Account,
		&hash.CreatedAt,
	)
	if err != nil {
		log.Println("Error retrieving data from HDB: ", err)
	}
	return &hash
}

func insertHashRecord(db *sql.DB, hashRecord HashRecord) error {
	_, err := db.Exec(
		insertHashSql,
		hashRecord.Id,
		hashRecord.CreatedAt,
		hashRecord.Key,
		hashRecord.HashToVerify,
		hashRecord.Account,
	)
	return err
}

func insertXuniRecord(db *sql.DB, hashRecord HashRecord) error {
	_, err := db.Exec(
		insertXuniSql,
		hashRecord.Id,
		hashRecord.CreatedAt,
		hashRecord.Key,
		hashRecord.HashToVerify,
		hashRecord.Account,
	)
	return err
}

func getAllBlocks(db *sql.DB) (c chan Block) {
	c = make(chan Block)
	rows, err := db.Query(getAllRowsBlockchainSql)
	if err != nil {
		log.Fatal("Error when querying DB: ", err)
	}
	go func() {
		defer func(rows *sql.Rows) {
			close(c)
			err := rows.Close()
			if err != nil {
				log.Println("Error when closing rows: ", err)
			}
		}(rows)
		var block Block
		for rows.Next() {
			_ = rows.Scan(&block.Id, &block.Timestamp, &block.PrevHash, &block.MerkleRoot, &block.RecordsJson, &block.BlockHash)
			c <- block
		}
	}()
	return c
}

func getHash(db *sql.DB, id uint) (*HashRecord, error) {
	row := db.QueryRow(getHashByIdSql, fmt.Sprintf("%d", id))
	var record HashRecord
	err := row.Scan(&record.Id, &record.CreatedAt, &record.Key, &record.HashToVerify, &record.Account)
	if err != nil {
		return nil, err
	}
	return &record, nil
}

func getXuni(db *sql.DB, id uint) (*HashRecord, error) {
	row := db.QueryRow(getXuniByIdSql, fmt.Sprintf("%d", id))
	var record HashRecord
	err := row.Scan(&record.Id, &record.CreatedAt, &record.Key, &record.HashToVerify, &record.Account)
	if err != nil {
		return nil, err
	}
	return &record, nil
}
