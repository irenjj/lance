// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! DataFusion execution utilities that only need DataFusion's sub-crates.
//!
//! The SQL planner in `lance-datafusion` re-exports these execution utilities.

pub mod chunker;
pub mod exec;
pub mod expr;
pub mod udf;
pub mod utils;
