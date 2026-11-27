// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Vector segment query contracts and validation before transaction publication.

use lance_core::{Error, Result};
use lance_index::{IndexMetadata, IndexType, vector::VectorIndex};
use lance_linalg::distance::DistanceType;

/// What a search ranks a vector segment by: its metric and its index type.
/// Merging the distances of segments that differ in either ranks part of the
/// data by the wrong one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorQueryContract {
    metric: DistanceType,
    /// `IndexType::Vector` for an index extension, which names no type this
    /// build knows; it matches only another such segment.
    index_type: IndexType,
}

impl VectorQueryContract {
    /// The contract of an opened segment, from the two accessors every
    /// vector index implements, legacy ones included.
    pub(crate) fn of_index(index: &dyn VectorIndex) -> Self {
        Self {
            metric: index.metric_type(),
            index_type: index.index_type(),
        }
    }

    /// The contract an index file of the current format records in its
    /// schema: the type the opener dispatches on, and the metric the opened
    /// index ranks by.
    pub(super) fn from_file_metadata(metadata: &IndexMetadata) -> Result<Self> {
        let index_type = IndexType::try_from(metadata.index_type.as_str())?;
        // The opener opens these files only as one of the IVF types.
        if !index_type.is_vector() || index_type == IndexType::Vector {
            return Err(Error::index(format!(
                "Unsupported index type: {}",
                metadata.index_type
            )));
        }
        Ok(Self {
            metric: DistanceType::try_from(metadata.distance_type.as_str())?,
            index_type,
        })
    }
}

impl std::fmt::Display for VectorQueryContract {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.index_type {
            // Its `Display` says IVF_PQ.
            IndexType::Vector => write!(f, "metric {:?}, an index extension's type", self.metric),
            index_type => write!(f, "metric {:?}, type {index_type}", self.metric),
        }
    }
}
