// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Dataset policies that differ across exact Lance file versions.
//!
//! File grammar belongs to `lance_file::versions`. This module contains only
//! operation-level dataset choices whose behavior actually differs by version.

use std::{collections::HashMap, ops::Range, sync::Arc};

use datafusion_catalog::Session;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::metrics::ExecutionPlanMetricsSet;
use lance_core::{
    Error, Result,
    datatypes::{Projection, Schema},
};
use lance_file::version::ConcreteFileVersion;
use lance_table::format::Fragment;

use super::Dataset;
use super::scanner::{PlannedFilteredScan, Scanner};
use super::statistics::FieldStatistics;
use crate::io::exec::{FilterPlan as ExprFilterPlan, LanceScanConfig, LanceStream};

#[allow(clippy::too_many_arguments)]
pub fn create_scan_stream(
    version: ConcreteFileVersion,
    dataset: Arc<Dataset>,
    fragments: Arc<Vec<Fragment>>,
    offsets: Option<Range<u64>>,
    projection: Arc<Schema>,
    config: LanceScanConfig,
    metrics: &ExecutionPlanMetricsSet,
    partition: usize,
) -> datafusion_common::Result<LanceStream> {
    match version {
        ConcreteFileVersion::V1 => LanceStream::try_new_v1(
            dataset, fragments, offsets, projection, config, metrics, partition,
        ),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => LanceStream::try_new_v2(
            dataset, fragments, offsets, projection, config, metrics, partition,
        ),
    }
}

#[allow(clippy::too_many_arguments)]
pub(in crate::dataset) async fn filtered_read(
    version: ConcreteFileVersion,
    scanner: &Scanner,
    filter_plan: &ExprFilterPlan,
    projection: Projection,
    make_deletions_null: bool,
    fragments: Option<Arc<Vec<Fragment>>>,
    scan_range: Option<Range<u64>>,
    is_prefilter: bool,
    session: Option<&dyn Session>,
) -> Result<PlannedFilteredScan> {
    match version {
        ConcreteFileVersion::V1 => {
            scanner
                .legacy_filtered_read(
                    filter_plan,
                    projection,
                    make_deletions_null,
                    fragments,
                    scan_range,
                    is_prefilter,
                )
                .await
        }
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            let limit_pushed_down = scan_range.is_some();
            let plan = scanner
                .new_filtered_read(
                    filter_plan,
                    projection,
                    make_deletions_null,
                    fragments,
                    scan_range,
                    session,
                )
                .await?;
            Ok(PlannedFilteredScan {
                filter_pushed_down: true,
                limit_pushed_down,
                plan,
            })
        }
    }
}

pub fn take(
    version: ConcreteFileVersion,
    scanner: &Scanner,
    input: Arc<dyn ExecutionPlan>,
    output_projection: Projection,
) -> Result<Arc<dyn ExecutionPlan>> {
    match version {
        ConcreteFileVersion::V1 => scanner.take_legacy(input, output_projection),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => scanner.take_current(input, output_projection),
    }
}

pub async fn collect_data_stats(
    version: ConcreteFileVersion,
    dataset: &Arc<Dataset>,
    field_stats: &mut HashMap<u32, FieldStatistics>,
) -> Result<()> {
    match version {
        ConcreteFileVersion::V1 => Ok(()),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => {
            super::statistics::collect_current_data_stats(dataset, field_stats).await
        }
    }
}

pub fn validate_row_stream_read(version: ConcreteFileVersion) -> Result<()> {
    match version {
        ConcreteFileVersion::V1 => Err(Error::not_supported_source(
            "taking rows through FilteredReadExec requires the v2 storage format"
                .to_string()
                .into(),
        )),
        ConcreteFileVersion::V2_0
        | ConcreteFileVersion::V2_1
        | ConcreteFileVersion::V2_2
        | ConcreteFileVersion::V2_3 => Ok(()),
    }
}
