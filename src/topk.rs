use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Index and similarity score returned by a nearest-neighbor search.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Neighbor {
    /// Row index in the stored matrix.
    pub index: usize,
    /// Cosine similarity score in `[-1.0, 1.0]` for normalized vectors.
    pub similarity: f32,
}

/// Internal heap entry pairing a row index with its similarity score.
///
/// Ordering is by similarity (ascending) with ties broken by index, so a min-heap
/// of `Reverse<HeapItem>` naturally surfaces the smallest score for eviction.
#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapItem {
    index: usize,
    similarity: f32,
}

impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.similarity
            .total_cmp(&other.similarity)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Bounded min-heap that retains only the k highest-scoring neighbors.
///
/// Uses a cached `min_threshold` for O(1) fast-reject of scores that cannot
/// enter the current top-k set, avoiding heap operations in the common case.
#[derive(Debug)]
pub(crate) struct TopKAccumulator {
    k: usize,
    heap: BinaryHeap<Reverse<HeapItem>>,
    /// Cached minimum similarity in the heap for O(1) fast-reject.
    min_threshold: f32,
}

impl TopKAccumulator {
    /// Create an empty accumulator that will retain at most `k` neighbors.
    pub(crate) fn new(k: usize) -> Self {
        Self {
            k,
            heap: BinaryHeap::with_capacity(k.saturating_add(1)),
            min_threshold: f32::NEG_INFINITY,
        }
    }

    /// Consider a candidate neighbor, inserting it only if it belongs in the top-k.
    #[inline]
    pub(crate) fn push(&mut self, index: usize, similarity: f32) {
        if self.heap.len() >= self.k {
            if similarity <= self.min_threshold {
                return;
            }
            let _ = self.heap.pop();
            self.heap.push(Reverse(HeapItem { index, similarity }));
            if let Some(smallest) = self.heap.peek() {
                self.min_threshold = smallest.0.similarity;
            }
            return;
        }

        self.heap.push(Reverse(HeapItem { index, similarity }));
        if self.heap.len() == self.k
            && let Some(smallest) = self.heap.peek()
        {
            self.min_threshold = smallest.0.similarity;
        }
    }

    /// Merge all entries from `other` into this accumulator, maintaining the top-k invariant.
    pub(crate) fn merge(&mut self, other: Self) {
        for item in other.heap {
            self.push(item.0.index, item.0.similarity);
        }
    }

    /// Consume the accumulator and return neighbors sorted by descending similarity.
    pub(crate) fn into_sorted_vec(self) -> Vec<Neighbor> {
        let mut out = self
            .heap
            .into_iter()
            .map(|item| Neighbor {
                index: item.0.index,
                similarity: item.0.similarity,
            })
            .collect::<Vec<_>>();
        out.sort_by(|left, right| {
            right
                .similarity
                .total_cmp(&left.similarity)
                .then_with(|| left.index.cmp(&right.index))
        });
        out.shrink_to_fit();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn retains_top_k_by_similarity() {
        let mut acc = TopKAccumulator::new(3);
        acc.push(0, 0.1);
        acc.push(1, 0.9);
        acc.push(2, 0.5);
        acc.push(3, 0.3);
        acc.push(4, 0.7);

        let result = acc.into_sorted_vec();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].index, 1);
        assert_eq!(result[1].index, 4);
        assert_eq!(result[2].index, 2);
    }

    #[test]
    fn fast_rejects_low_scores_after_filling() {
        let mut acc = TopKAccumulator::new(2);
        acc.push(0, 0.8);
        acc.push(1, 0.9);
        // These should be fast-rejected without entering the heap.
        acc.push(2, 0.1);
        acc.push(3, 0.0);
        acc.push(4, -1.0);

        let result = acc.into_sorted_vec();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].index, 1);
        assert_eq!(result[1].index, 0);
    }

    #[test]
    fn merge_combines_two_accumulators() {
        let mut left = TopKAccumulator::new(3);
        left.push(0, 0.9);
        left.push(1, 0.1);
        left.push(2, 0.5);

        let mut right = TopKAccumulator::new(3);
        right.push(3, 0.8);
        right.push(4, 0.2);
        right.push(5, 0.7);

        left.merge(right);
        let result = left.into_sorted_vec();
        assert_eq!(result.len(), 3);
        assert_eq!(result[0].index, 0);
        assert_eq!(result[1].index, 3);
        assert_eq!(result[2].index, 5);
    }

    #[test]
    fn ties_broken_by_lower_index_first_in_output() {
        let mut acc = TopKAccumulator::new(3);
        acc.push(5, 0.5);
        acc.push(2, 0.5);
        acc.push(8, 0.5);

        let result = acc.into_sorted_vec();
        assert_eq!(result.len(), 3);
        // With identical scores, output is sorted with lower indices first.
        assert_eq!(result[0].index, 2);
        assert_eq!(result[1].index, 5);
        assert_eq!(result[2].index, 8);
    }

    #[test]
    fn equal_score_rejected_when_heap_full() {
        // The fast-reject threshold uses <=, so equal scores don't displace existing entries.
        let mut acc = TopKAccumulator::new(2);
        acc.push(0, 0.5);
        acc.push(1, 0.5);
        acc.push(2, 0.5);

        let result = acc.into_sorted_vec();
        assert_eq!(result.len(), 2);
        // Original two entries kept; third was rejected.
        let indices: Vec<usize> = result.iter().map(|n| n.index).collect();
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }

    #[test]
    fn fewer_items_than_k_returns_all() {
        let mut acc = TopKAccumulator::new(10);
        acc.push(0, 0.3);
        acc.push(1, 0.7);

        let result = acc.into_sorted_vec();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].index, 1);
        assert_eq!(result[1].index, 0);
    }

    #[test]
    fn k_equals_one_returns_single_best() {
        let mut acc = TopKAccumulator::new(1);
        acc.push(0, 0.1);
        acc.push(1, 0.9);
        acc.push(2, 0.5);

        let result = acc.into_sorted_vec();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].index, 1);
        assert_eq!(result[0].similarity, 0.9);
    }

    #[test]
    fn negative_similarities_handled_correctly() {
        let mut acc = TopKAccumulator::new(2);
        acc.push(0, -0.9);
        acc.push(1, -0.1);
        acc.push(2, -0.5);

        let result = acc.into_sorted_vec();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].index, 1);
        assert_eq!(result[1].index, 2);
    }
}
