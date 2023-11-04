package main

const (
	createControlTableSql string = `
			CREATE TABLE IF NOT EXISTS control (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            node TEXT,
            blocks_range TEXT,
            hash TEXT,
            difficulty INT,
            ts INTEGER);
			CREATE UNIQUE INDEX IF NOT EXISTS idx_node_blocks_range ON control (node, blocks_range);
	`

	insertRangeSql = `
		INSERT INTO control (node, blocks_range, hash, difficulty, ts)
		VALUES (?, ?, ?, ?, ?);
	`

	createBlockchainTableSql string = `
		CREATE TABLE IF NOT EXISTS blockchain (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
		prev_hash TEXT,
		merkle_root TEXT,
		records_json TEXT,
		block_hash TEXT)
	`

	getMaxHeightBlockchainSql string = `
		SELECT MAX(id) as max_height 
		FROM blockchain;
	`

	insertBlockchainSql = `
    	INSERT INTO blockchain (id, timestamp, prev_hash, merkle_root, records_json, block_hash)
     	VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT DO NOTHING;
	`

	getRowBlockchainSql string = `
		SELECT id, timestamp, prev_hash, merkle_root, records_json, block_hash 
		FROM blockchain 
		WHERE id = ?;
	`

	getAllRowsBlockchainSql string = `
		SELECT id, timestamp, prev_hash, merkle_root, records_json, block_hash 
		FROM blockchain 
		ORDER BY id DESC;
	`

	getMissingRowIdsBlockchainSql = `
		select id+1 from blockchain bo 
		where not exists (
			select null 
			from blockchain bi 
			where bi.id = bo.id + 1
		) group by id 
		limit 10	
	`

	getMissingHashRowIdsSql = `
		select block_id+1 from blocks bo                 
		where not exists (
			select null 
			from blocks bi 
			where bi.block_id = bo.block_id + 1
			order by block_id asc
		) 
		group by block_id 
		order by block_id asc 
		limit 10	
	`

	getMissingXuniRowIdsSql = `
		select id+1 from xuni xo 
		where not exists (
			select null 
			from xuni xi 
			where xi.id = xo.id + 1
			order by id asc
		) 
		group by id 
		order by id asc 
		limit 10	
	`

	initDbSql = `VACUUM;`

	resetDbSql = `delete from blockchain;`

	resetHashesSql = `delete from blocks;`

	resetXunisSql = `delete from xuni;`

	createHashesTableSql = `
		CREATE TABLE IF NOT EXISTS blocks (
    		block_id INTEGER PRIMARY KEY,
    		hash_to_verify TEXT,
    		key TEXT UNIQUE,
    		account TEXT,
    		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);
	`

	createXunisTableSql = `
		CREATE TABLE IF NOT EXISTS xuni (
    		id INTEGER PRIMARY KEY,
    		hash_to_verify TEXT,
    		key TEXT UNIQUE,
    		account TEXT,
    		created_at DATETIME DEFAULT CURRENT_TIMESTAMP
		);
	`

	insertHashSql = `
		INSERT INTO blocks (block_id, created_at, key, hash_to_verify, account)
		VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING;
	`

	insertXuniSql = `
		INSERT INTO xuni (id, created_at, key, hash_to_verify, account)
		VALUES (?, ?, ?, ?, ?) ON CONFLICT DO NOTHING;
	`
	getLatestHashIdSql string = `
		SELECT MAX(block_id) as latest_hash_id 
		FROM blocks;
	`

	getLatestXuniIdSql string = `
		SELECT MAX(id) as latest_xuni_id 
		FROM xuni;
	`

	getLatestHashSql string = `
		SELECT block_id, hash_to_verify, key, account, created_at 
		FROM blocks 
		ORDER BY block_id DESC 
		LIMIT 1;
	`

	getLatestRangeSql string = `
		SELECT id, node, blocks_range, hash, difficulty, ts
		FROM control 
		WHERE node = 'myself'
		ORDER BY id DESC 
		LIMIT 1;
	`

	getLatestXuniSql string = `
		SELECT id, hash_to_verify, key, account, created_at 
		FROM xuni 
		ORDER BY id DESC 
		LIMIT 1;
	`

	getHashByIdSql string = `
		SELECT block_id, hash_to_verify, key, account, created_at 
		FROM blocks 
		WHERE block_id = ?;
	`

	getXuniByIdSql string = `
		SELECT id, hash_to_verify, key, account, created_at  
		FROM xuni 
		WHERE id = ?;
	`
)
