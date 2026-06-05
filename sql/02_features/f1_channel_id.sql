-- f1_channel_id: pass-through of the channel column (ad publisher channel).
-- Contract: returns (row_id, value), one row per {source} row.
SELECT row_id, channel AS value FROM {source}
