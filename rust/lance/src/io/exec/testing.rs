// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Testing Node
//!

use std::sync::Arc;

use arrow_array::RecordBatch;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_execution::TaskContext;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning};
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
    execution_plan::{Boundedness, EmissionType},
};
use futures::StreamExt;

#[derive(Debug)]
pub struct TestingExec {
    pub(crate) batches: Vec<RecordBatch>,
    properties: Arc<PlanProperties>,
}

#[cfg(feature = "write")]
pub(crate) fn new(batches: Vec<RecordBatch>) -> Self {
    let properties = Arc::new(PlanProperties::new(
        EquivalenceProperties::new(batches[0].schema()),
        Partitioning::RoundRobinBatch(1),
        EmissionType::Incremental,
        Boundedness::Bounded,
    ));
    Self {
        batches,
        properties,
    }
}

impl TestingExec {}

impl DisplayAs for TestingExec {
    fn fmt_as(&self, _t: DisplayFormatType, _f: &mut std::fmt::Formatter) -> std::fmt::Result {
        todo!()
    }
}

impl ExecutionPlan for TestingExec {
    fn name(&self) -> &str {
        "TestingExec"
    }

    fn schema(&self) -> arrow_schema::SchemaRef {
        self.batches[0].schema()
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> datafusion_common::Result<TreeNodeRecursion>,
    ) -> datafusion_common::Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn with_new_children(
        self: Arc<Self>,
        _children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> datafusion_common::Result<std::sync::Arc<dyn ExecutionPlan>> {
        todo!()
    }

    fn execute(
        &self,
        _partition: usize,
        _context: Arc<TaskContext>,
    ) -> datafusion_common::Result<SendableRecordBatchStream> {
        let stream = futures::stream::iter(self.batches.clone().into_iter().map(Ok));
        let stream = RecordBatchStreamAdapter::new(self.schema(), stream.boxed());
        Ok(Box::pin(stream))
    }

    fn properties(&self) -> &Arc<datafusion_physical_plan::PlanProperties> {
        &self.properties
    }
}
