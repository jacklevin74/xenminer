-- Add the new block and xuni IDs to allblocks2.
-- New rows always get a higher ID, so only IDs above the current maximum are new.
-- After rows are inserted with explicit IDs, run allblocks2-full.sql once.
.timeout 30000
BEGIN TRANSACTION;

INSERT INTO allblocks2 (block_id)
SELECT block_id FROM blocks
WHERE block_id > (SELECT IFNULL(MAX(block_id), 0) FROM allblocks2);

INSERT INTO allblocks2 (xuni_id)
SELECT Id FROM xuni
WHERE Id > (SELECT IFNULL(MAX(xuni_id), 0) FROM allblocks2);

COMMIT;
