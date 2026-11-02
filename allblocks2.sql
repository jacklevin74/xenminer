BEGIN TRANSACTION;

-- Attempt to insert into block_id only if it doesn't exist in allblocks2
INSERT INTO allblocks2 (block_id)
SELECT block_id FROM blocks
WHERE NOT EXISTS (
    SELECT 1 FROM allblocks2 WHERE allblocks2.block_id = blocks.block_id
);

-- Attempt to insert into xuni_id only if it doesn't exist in allblocks2
INSERT INTO allblocks2 (xuni_id)
SELECT Id FROM xuni
WHERE NOT EXISTS (
    SELECT 1 FROM allblocks2 WHERE allblocks2.xuni_id = xuni.Id
);

COMMIT;

