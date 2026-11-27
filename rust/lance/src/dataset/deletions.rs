// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Deletion vectors stored in the manifest instead of in `_deletions/` files.

use lance_core::Result;

use crate::Dataset;

pub use lance_table_runtime::io::deletion::INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY;

impl Dataset {
    /// The inline budget from [`INLINE_DELETIONS_MAX_BYTES_CONFIG_KEY`], `0` when unset.
    pub fn inline_deletions_max_bytes(&self) -> Result<usize> {
        lance_table_runtime::io::deletion::inline_deletions_max_bytes(&self.manifest)
    }
}
