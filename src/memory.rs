// Persistent episodic memory -- T-019
//
// SQLite-backed: concept_vec, goal, response, success, timestamp
// Methods: store(), retrieve_top_k(), consolidate()
// Wraps brain.rs MemoryBank pattern with disk persistence.

use anyhow::Result;
use rusqlite::{params, Connection};

/// Persistent episodic memory backed by SQLite.
/// Stores (concept_vec, goal, response, success) tuples.
/// Retrieves top-K by cosine similarity on concept_vec.
pub struct EpisodicMemory {
    conn: Connection,
    capacity: usize,
    d_model: usize,
}

/// A single retrieved memory entry.
#[derive(Debug, Clone)]
pub struct MemoryRecord {
    pub id: i64,
    pub goal: String,
    pub response: String,
    pub success: bool,
    pub concept_vec: Vec<f32>,
    pub timestamp: String,
}

impl EpisodicMemory {
    /// Open (or create) a memory database at the given path.
    /// Use ":memory:" for in-memory testing.
    pub fn open(path: &str, d_model: usize, capacity: usize) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL DEFAULT (datetime('now')),
                concept_vec BLOB NOT NULL,
                goal TEXT NOT NULL,
                response TEXT NOT NULL,
                success INTEGER NOT NULL DEFAULT 1
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_ts ON episodes(timestamp);",
        )?;
        Ok(Self { conn, capacity, d_model })
    }

    /// Store an episodic memory. Evicts oldest if at capacity.
    pub fn store(
        &self,
        concept_vec: &[f32],
        goal: &str,
        response: &str,
        success: bool,
    ) -> Result<i64> {
        assert_eq!(concept_vec.len(), self.d_model, "concept_vec dim mismatch");

        // FIFO eviction
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM episodes", [], |r| r.get(0),
        )?;
        if count >= self.capacity as i64 {
            let excess = count - self.capacity as i64 + 1;
            self.conn.execute(
                "DELETE FROM episodes WHERE id IN (
                    SELECT id FROM episodes ORDER BY id ASC LIMIT ?1
                )",
                params![excess],
            )?;
        }

        let blob = vec_to_blob(concept_vec);
        self.conn.execute(
            "INSERT INTO episodes (concept_vec, goal, response, success) VALUES (?1, ?2, ?3, ?4)",
            params![blob, goal, response, success as i32],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Retrieve top-K memories by cosine similarity to the query vector.
    /// Returns records sorted by similarity (highest first).
    pub fn retrieve_top_k(&self, query: &[f32], k: usize) -> Result<Vec<MemoryRecord>> {
        assert_eq!(query.len(), self.d_model, "query dim mismatch");

        let q_norm = l2_norm(query);
        if q_norm < 1e-8 {
            return Ok(vec![]);
        }

        let mut stmt = self.conn.prepare(
            "SELECT id, timestamp, concept_vec, goal, response, success FROM episodes"
        )?;

        let mut scored: Vec<(f32, MemoryRecord)> = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let timestamp: String = row.get(1)?;
            let blob: Vec<u8> = row.get(2)?;
            let goal: String = row.get(3)?;
            let response: String = row.get(4)?;
            let success: i32 = row.get(5)?;
            Ok((id, timestamp, blob, goal, response, success != 0))
        })?
        .filter_map(|r| r.ok())
        .map(|(id, timestamp, blob, goal, response, success)| {
            let cv = blob_to_vec(&blob);
            let e_norm = l2_norm(&cv);
            let sim = if e_norm > 1e-8 {
                dot(&cv, query) / (q_norm * e_norm)
            } else {
                0.0
            };
            (sim, MemoryRecord { id, goal, response, success, concept_vec: cv, timestamp })
        })
        .collect();

        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        Ok(scored.into_iter().map(|(_, r)| r).collect())
    }

    /// Retrieve concept vectors only (for brain prefix building).
    /// Returns Vec of concept_vec slices, highest similarity first.
    pub fn retrieve_vecs(&self, query: &[f32], k: usize) -> Result<Vec<Vec<f32>>> {
        let records = self.retrieve_top_k(query, k)?;
        Ok(records.into_iter().map(|r| r.concept_vec).collect())
    }

    /// Number of stored episodes.
    pub fn len(&self) -> Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM episodes", [], |r| r.get(0),
        )?;
        Ok(count as usize)
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Delete all episodes. Used for testing.
    pub fn clear(&self) -> Result<()> {
        self.conn.execute("DELETE FROM episodes", [])?;
        Ok(())
    }

    /// Consolidate: merge similar old memories to reduce count.
    /// Groups memories by cosine similarity > threshold,
    /// keeps the most recent from each group.
    pub fn consolidate(&self, similarity_threshold: f32) -> Result<usize> {
        let mut stmt = self.conn.prepare(
            "SELECT id, concept_vec FROM episodes ORDER BY id ASC"
        )?;
        let entries: Vec<(i64, Vec<f32>)> = stmt.query_map([], |row| {
            let id: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((id, blob_to_vec(&blob)))
        })?.filter_map(|r| r.ok()).collect();

        if entries.len() < 2 {
            return Ok(0);
        }

        let mut to_delete: Vec<i64> = Vec::new();
        let mut consumed: Vec<bool> = vec![false; entries.len()];

        for i in 0..entries.len() {
            if consumed[i] { continue; }
            let i_norm = l2_norm(&entries[i].1);
            if i_norm < 1e-8 { continue; }

            for j in (i + 1)..entries.len() {
                if consumed[j] { continue; }
                let j_norm = l2_norm(&entries[j].1);
                if j_norm < 1e-8 { continue; }

                let sim = dot(&entries[i].1, &entries[j].1) / (i_norm * j_norm);
                if sim >= similarity_threshold {
                    // Keep newer (j), delete older (i)
                    to_delete.push(entries[i].0);
                    consumed[i] = true;
                    break;
                }
            }
        }

        for id in &to_delete {
            self.conn.execute("DELETE FROM episodes WHERE id = ?1", params![id])?;
        }
        Ok(to_delete.len())
    }
}

// --- Serialization helpers ---

fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn l2_norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn test_mem(capacity: usize) -> EpisodicMemory {
        EpisodicMemory::open(":memory:", 3, capacity).unwrap()
    }

    #[test]
    fn test_store_and_retrieve() -> Result<()> {
        let mem = test_mem(100);
        mem.store(&[1.0, 0.0, 0.0], "hello", "Hi there!", true)?;
        mem.store(&[0.0, 1.0, 0.0], "test code", "Running tests...", true)?;
        mem.store(&[0.9, 0.1, 0.0], "hey", "Hey, what's up?", true)?;

        let results = mem.retrieve_top_k(&[1.0, 0.0, 0.0], 2)?;
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].goal, "hello"); // exact match
        assert_eq!(results[1].goal, "hey");   // close match
        Ok(())
    }

    #[test]
    fn test_retrieve_vecs() -> Result<()> {
        let mem = test_mem(100);
        mem.store(&[1.0, 0.0, 0.0], "a", "resp_a", true)?;
        mem.store(&[0.0, 1.0, 0.0], "b", "resp_b", true)?;

        let vecs = mem.retrieve_vecs(&[1.0, 0.0, 0.0], 1)?;
        assert_eq!(vecs.len(), 1);
        assert!((vecs[0][0] - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_fifo_eviction() -> Result<()> {
        let mem = test_mem(3);
        mem.store(&[1.0, 0.0, 0.0], "first", "r1", true)?;
        mem.store(&[0.0, 1.0, 0.0], "second", "r2", true)?;
        mem.store(&[0.0, 0.0, 1.0], "third", "r3", true)?;
        assert_eq!(mem.len()?, 3);

        mem.store(&[0.5, 0.5, 0.0], "fourth", "r4", true)?;
        assert_eq!(mem.len()?, 3);

        // "first" should be evicted
        let all = mem.retrieve_top_k(&[1.0, 1.0, 1.0], 10)?;
        assert!(!all.iter().any(|r| r.goal == "first"));
        assert!(all.iter().any(|r| r.goal == "fourth"));
        Ok(())
    }

    #[test]
    fn test_persistence_roundtrip() -> Result<()> {
        let path = "/tmp/gestalt_test_memory.db";
        // Clean up
        let _ = std::fs::remove_file(path);

        {
            let mem = EpisodicMemory::open(path, 4, 100)?;
            mem.store(&[1.0, 2.0, 3.0, 4.0], "remember this", "blue", true)?;
        }
        // Reopen
        {
            let mem = EpisodicMemory::open(path, 4, 100)?;
            assert_eq!(mem.len()?, 1);
            let results = mem.retrieve_top_k(&[1.0, 2.0, 3.0, 4.0], 1)?;
            assert_eq!(results[0].goal, "remember this");
            assert_eq!(results[0].response, "blue");
        }
        let _ = std::fs::remove_file(path);
        Ok(())
    }

    #[test]
    fn test_empty_retrieval() -> Result<()> {
        let mem = test_mem(100);
        let results = mem.retrieve_top_k(&[1.0, 0.0, 0.0], 5)?;
        assert!(results.is_empty());
        Ok(())
    }

    #[test]
    fn test_zero_query() -> Result<()> {
        let mem = test_mem(100);
        mem.store(&[1.0, 0.0, 0.0], "test", "resp", true)?;
        let results = mem.retrieve_top_k(&[0.0, 0.0, 0.0], 5)?;
        assert!(results.is_empty()); // zero query returns nothing
        Ok(())
    }

    #[test]
    fn test_consolidate() -> Result<()> {
        let mem = test_mem(100);
        // Store two very similar memories
        mem.store(&[1.0, 0.0, 0.0], "hello", "Hi!", true)?;
        mem.store(&[0.99, 0.01, 0.0], "hey", "Hey!", true)?;
        // And one different
        mem.store(&[0.0, 1.0, 0.0], "test", "Testing...", true)?;

        assert_eq!(mem.len()?, 3);
        let removed = mem.consolidate(0.95)?;
        assert_eq!(removed, 1); // "hello" merged into "hey"
        assert_eq!(mem.len()?, 2);

        // "hey" (newer) should survive, "hello" (older) removed
        let all = mem.retrieve_top_k(&[1.0, 1.0, 1.0], 10)?;
        assert!(all.iter().any(|r| r.goal == "hey"));
        assert!(!all.iter().any(|r| r.goal == "hello"));
        Ok(())
    }

    #[test]
    fn test_success_flag() -> Result<()> {
        let mem = test_mem(100);
        mem.store(&[1.0, 0.0, 0.0], "worked", "yes", true)?;
        mem.store(&[0.0, 1.0, 0.0], "failed", "no", false)?;

        let results = mem.retrieve_top_k(&[0.0, 1.0, 0.0], 1)?;
        assert!(!results[0].success);
        Ok(())
    }
}
