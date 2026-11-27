// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::sync::Arc;

use arrow_schema::{Schema, SchemaRef};
use async_trait::async_trait;
use datafusion_catalog::Session;
use datafusion_catalog::TableProvider;
use datafusion_common::DataFusionError;
use datafusion_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion_physical_plan::ExecutionPlan;
use lance_arrow::SchemaExt;
#[cfg(test)]
use lance_core::ROW_CREATED_AT_VERSION_FIELD;
use lance_core::{ROW_ADDR_FIELD, ROW_ID_FIELD};

use crate::Dataset;

/// A [TableProvider] for Lance datasets.
///
/// Note: Datafusion has no concept of "system columns".  As a result, you must specify
/// which schema columns should be included in the table's schema when you create the
/// provider.
///
/// This table provider should support:
///  - Filter pushdown
///  - Limit pushdown
///  - Projection pushdown
///
/// Note that LanceDB also has a TableProvider implementation that should be preferred
/// if you are working in LanceDB.
#[derive(Debug)]
pub struct LanceTableProvider {
    dataset: Arc<Dataset>,
    full_schema: Arc<Schema>,
    row_id_idx: Option<usize>,
    row_addr_idx: Option<usize>,
    row_created_at_idx: Option<usize>,
    ordered: bool,
    blob_handling: Option<lance_core::datatypes::BlobHandling>,
    batch_size: Option<usize>,
    batch_size_bytes: Option<u64>,
}

impl LanceTableProvider {
    pub fn new(dataset: Arc<Dataset>, with_row_id: bool, with_row_addr: bool) -> Self {
        Self::new_with_ordering(dataset, with_row_id, with_row_addr, true)
    }

    pub fn new_with_ordering(
        dataset: Arc<Dataset>,
        with_row_id: bool,
        with_row_addr: bool,
        ordered: bool,
    ) -> Self {
        let mut full_schema = Schema::from(dataset.schema());
        let mut row_id_idx = None;
        let mut row_addr_idx = None;
        if with_row_id {
            full_schema = full_schema.try_with_column(ROW_ID_FIELD.clone()).unwrap();
            row_id_idx = Some(full_schema.fields.len() - 1);
        }
        if with_row_addr {
            full_schema = full_schema.try_with_column(ROW_ADDR_FIELD.clone()).unwrap();
            row_addr_idx = Some(full_schema.fields.len() - 1);
        }
        Self {
            dataset,
            full_schema: Arc::new(full_schema),
            row_id_idx,
            row_addr_idx,
            row_created_at_idx: None,
            ordered,
            blob_handling: None,
            batch_size: None,
            batch_size_bytes: None,
        }
    }

    /// Overrides how blob columns are read during [`TableProvider::scan`].
    /// When unset, the underlying dataset scan uses its default
    /// [`BlobHandling`](lance_core::datatypes::BlobHandling) policy.
    pub fn with_blob_handling(mut self, handling: lance_core::datatypes::BlobHandling) -> Self {
        let converted = self
            .dataset
            .full_projection()
            .with_blob_handling(handling.clone())
            .to_bare_schema();
        let mut full_schema = Schema::from(&converted);
        if self.row_id_idx.is_some() {
            full_schema = full_schema.try_with_column(ROW_ID_FIELD.clone()).unwrap();
        }
        if self.row_addr_idx.is_some() {
            full_schema = full_schema.try_with_column(ROW_ADDR_FIELD.clone()).unwrap();
        }
        if let Some(idx) = self.row_created_at_idx {
            full_schema.fields = full_schema
                .fields
                .iter()
                .cloned()
                .chain([self.full_schema.fields[idx].clone()])
                .collect();
        }
        self.full_schema = Arc::new(full_schema);
        self.blob_handling = Some(handling);
        self
    }

    #[cfg(test)]
    /// Exposes the target rows' creation versions for rewrite operations that
    /// carry their lineage into new fragments.
    pub(crate) fn with_row_created_at_version(mut self) -> crate::Result<Self> {
        if self.row_created_at_idx.is_none() {
            let schema = self
                .full_schema
                .try_with_column(ROW_CREATED_AT_VERSION_FIELD.clone())?;
            self.row_created_at_idx = Some(schema.fields.len() - 1);
            self.full_schema = Arc::new(schema);
        }
        Ok(self)
    }

    /// Overrides the maximum number of rows produced by each dataset scan batch.
    ///
    /// The batch size must be between 1 and [`u32::MAX`], inclusive. Invalid
    /// values are rejected when DataFusion creates the scan plan.
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Overrides the approximate maximum bytes produced by each dataset scan batch.
    pub fn with_batch_size_bytes(mut self, batch_size_bytes: u64) -> Self {
        self.batch_size_bytes = Some(batch_size_bytes);
        self
    }

    pub fn dataset(&self) -> Arc<Dataset> {
        self.dataset.clone()
    }
}

#[async_trait]
impl TableProvider for LanceTableProvider {
    fn schema(&self) -> SchemaRef {
        self.full_schema.clone()
    }

    fn table_type(&self) -> TableType {
        TableType::Base
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion_common::Result<Arc<dyn ExecutionPlan>> {
        let mut scan = self.dataset.scan();
        if let Some(handling) = self.blob_handling.clone() {
            scan.blob_handling(handling);
        }
        if let Some(batch_size) = self.batch_size {
            scan.batch_size(batch_size);
        }
        if let Some(batch_size_bytes) = self.batch_size_bytes {
            scan.batch_size_bytes(batch_size_bytes);
        }

        let projected = match projection {
            Some(indices) => Arc::new(self.full_schema.project(indices)?),
            None => self.full_schema.clone(),
        };
        let columns = projected
            .fields()
            .iter()
            .map(|field| field.name())
            .collect::<Vec<_>>();
        scan.project(&columns)?;

        let combined_filter = match filters.len() {
            0 => None,
            1 => Some(filters[0].clone()),
            _ => {
                let mut expr = filters[0].clone();
                for filter in &filters[1..] {
                    expr = Expr::and(expr, filter.clone());
                }
                Some(expr)
            }
        };
        if let Some(combined_filter) = combined_filter {
            scan.filter_expr(combined_filter);
        }
        scan.limit(limit.map(|l| l as i64), None)?;
        scan.scan_in_order(self.ordered);

        scan.create_plan_with_session(state)
            .await
            .map_err(DataFusionError::from)
    }

    // Since we are using datafusion itself to apply the filters it should
    // be safe to assume that we can exactly apply any of the given pushdown
    // filters.
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion_common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|_| TableProviderFilterPushDown::Exact)
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::{Dataset, datafusion::LanceTableProvider};
    use arrow::{array::AsArray, datatypes::UInt64Type};
    use arrow_array::record_batch;
    use datafusion_catalog::TableProvider;
    use datafusion_physical_plan::collect;
    use datafusion_session::Session;
    use lance_core::datatypes::BlobHandling;
    use lance_table_runtime::write::{InsertBuilder, WriteParams};
    use lance_testing::datafusion::TestSession;

    use lance_core::{ROW_ADDR, ROW_CREATED_AT_VERSION, ROW_ID};
    use rstest::rstest;

    #[rstest]
    #[case::all(None)]
    #[case::schema_order(Some(vec![0, 1, 2, 3]))]
    #[case::permuted(Some(vec![3, 1, 0, 2]))]
    #[case::lineage_only(Some(vec![1, 3]))]
    #[tokio::test]
    async fn table_provider_created_at_projection_preserves_order(
        #[case] projection: Option<Vec<usize>>,
    ) {
        let batch = record_batch!(("key", UInt32, [10, 11])).unwrap();
        let table = InsertBuilder::new("memory://")
            .with_params(&WriteParams {
                enable_stable_row_ids: true,
                max_rows_per_file: 1,
                ..Default::default()
            })
            .execute(vec![batch])
            .await
            .unwrap();
        let dataset = Dataset::from_table(table, Arc::new(crate::session::Session::default()));
        let provider = LanceTableProvider::new(Arc::new(dataset), true, true)
            .with_row_created_at_version()
            .unwrap()
            .with_blob_handling(BlobHandling::default());
        let context = TestSession::default();
        let plan = provider
            .scan(&context, projection.as_deref(), &[], None)
            .await
            .unwrap();
        let expected_schema = match &projection {
            Some(projection) => provider.schema().project(projection).unwrap(),
            None => provider.schema().as_ref().clone(),
        };
        assert_eq!(plan.schema().fields(), expected_schema.fields());
        let batches = collect(plan, context.task_ctx()).await.unwrap();
        let batch =
            arrow_select::concat::concat_batches(&Arc::new(expected_schema), &batches).unwrap();
        assert_eq!(batch.num_rows(), 2);
        for (name, expected) in [
            (ROW_ID, vec![0, 1]),
            (ROW_ADDR, vec![0, 1_u64 << 32]),
            (ROW_CREATED_AT_VERSION, vec![1, 1]),
        ] {
            if let Some(column) = batch.column_by_name(name) {
                assert_eq!(
                    column.as_primitive::<UInt64Type>().values().as_ref(),
                    expected
                );
            }
        }
    }

    #[tokio::test]
    async fn test_table_provider_rejects_invalid_batch_size() {
        let batch = record_batch!(("x", Int32, [1, 2, 3])).unwrap();
        let table = InsertBuilder::new("memory://")
            .execute(vec![batch])
            .await
            .unwrap();
        let data = Arc::new(Dataset::from_table(
            table,
            Arc::new(crate::session::Session::default()),
        ));
        let session = TestSession::default();

        for batch_size in [0, u32::MAX as usize + 1] {
            let provider =
                LanceTableProvider::new(data.clone(), false, false).with_batch_size(batch_size);
            let error = provider
                .scan(&session, None, &[], None)
                .await
                .expect_err("invalid batch size should be rejected");
            assert!(
                error
                    .to_string()
                    .contains(&format!("batch_size must be between 1 and {}", u32::MAX))
            );
        }
    }
}
