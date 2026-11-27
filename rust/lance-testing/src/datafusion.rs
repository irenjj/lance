// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Modular DataFusion session for testing table providers.

use std::{any::Any, collections::HashMap, sync::Arc};

use async_trait::async_trait;
use datafusion_catalog::MemoryCatalogProviderList;
use datafusion_common::{DFSchema, DataFusionError, Result, config::TableOptions};
use datafusion_execution::{TaskContext, config::SessionConfig, runtime_env::RuntimeEnv};
use datafusion_expr::{
    AggregateUDF, Expr, HigherOrderUDF, LogicalPlan, ScalarUDF, WindowUDF,
    execution_props::ExecutionProps,
    physical_planning_context::PhysicalPlanningContext,
    registry::{ExtensionTypeRegistryRef, MemoryExtensionTypeRegistry},
};
use datafusion_physical_expr::{PhysicalExpr, create_physical_expr};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_session::{CatalogProviderList, Session};

/// Use a real task context with provider APIs that take a modular session.
pub struct TestSession {
    id: String,
    task: Arc<TaskContext>,
    runtime: Arc<RuntimeEnv>,
    catalogs: Arc<dyn CatalogProviderList>,
    properties: ExecutionProps,
    extension_types: ExtensionTypeRegistryRef,
    table_options: TableOptions,
}

impl TestSession {
    pub fn new(task: Arc<TaskContext>) -> Self {
        Self {
            id: task.session_id(),
            runtime: task.runtime_env(),
            task,
            catalogs: Arc::new(MemoryCatalogProviderList::new()),
            properties: ExecutionProps::new(),
            extension_types: Arc::new(MemoryExtensionTypeRegistry::default()),
            table_options: TableOptions::default(),
        }
    }
}

impl Default for TestSession {
    fn default() -> Self {
        Self::new(Arc::new(TaskContext::default()))
    }
}

#[async_trait]
impl Session for TestSession {
    fn session_id(&self) -> &str {
        &self.id
    }

    fn config(&self) -> &SessionConfig {
        self.task.session_config()
    }

    fn catalog_list(&self) -> Arc<dyn CatalogProviderList> {
        self.catalogs.clone()
    }

    async fn create_physical_plan(&self, _: &LogicalPlan) -> Result<Arc<dyn ExecutionPlan>> {
        Err(DataFusionError::NotImplemented(
            "TestSession executes provider scans directly".into(),
        ))
    }

    fn create_physical_expr(&self, expr: Expr, schema: &DFSchema) -> Result<Arc<dyn PhysicalExpr>> {
        create_physical_expr(
            &expr,
            schema,
            &self.properties,
            &PhysicalPlanningContext::default(),
        )
    }

    fn scalar_functions(&self) -> &HashMap<String, Arc<ScalarUDF>> {
        self.task.scalar_functions()
    }

    fn higher_order_functions(&self) -> &HashMap<String, Arc<HigherOrderUDF>> {
        self.task.higher_order_functions()
    }

    fn aggregate_functions(&self) -> &HashMap<String, Arc<AggregateUDF>> {
        self.task.aggregate_functions()
    }

    fn window_functions(&self) -> &HashMap<String, Arc<WindowUDF>> {
        self.task.window_functions()
    }

    fn extension_type_registry(&self) -> &ExtensionTypeRegistryRef {
        &self.extension_types
    }

    fn runtime_env(&self) -> &Arc<RuntimeEnv> {
        &self.runtime
    }

    fn execution_props(&self) -> &ExecutionProps {
        &self.properties
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn table_options(&self) -> &TableOptions {
        &self.table_options
    }

    fn table_options_mut(&mut self) -> &mut TableOptions {
        &mut self.table_options
    }

    fn task_ctx(&self) -> Arc<TaskContext> {
        self.task.clone()
    }
}
