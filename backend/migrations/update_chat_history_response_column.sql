-- Migration script to update chat_history.response column from VARCHAR(1000) to TEXT
-- Run this SQL script in your MySQL database to fix the "Data too long" error

-- For MySQL/MariaDB:
ALTER TABLE chat_history MODIFY COLUMN response TEXT NOT NULL;

-- If the above doesn't work, try:
-- ALTER TABLE chat_history CHANGE response response TEXT NOT NULL;

-- Verify the change:
-- DESCRIBE chat_history;



