// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use async_trait::async_trait;
use lance_table::format::IndexMetadata;
use lance_table_runtime::{Table, commit::LegacyIndexDetails};

use super::vector::details::{infer_missing_vector_details, needs_vector_details_inference};
use crate::{Dataset, session::Session};

pub(super) struct LegacyVectorDetails<'a>(pub &'a Arc<Session>);

#[async_trait]
impl LegacyIndexDetails for LegacyVectorDetails<'_> {
    fn needs_legacy_details(&self, table: &Table, index: &IndexMetadata) -> bool {
        needs_vector_details_inference(index, table.schema())
    }

    fn legacy_details_cache_namespace(&self) -> Option<&'static str> {
        Some("lance.vector-details")
    }

    async fn infer_legacy_details(&self, table: &Table, indices: &mut [IndexMetadata]) {
        let dataset = Dataset::from_committed_table(table.clone(), self.0.clone());
        infer_missing_vector_details(&dataset, indices).await;
    }
}
