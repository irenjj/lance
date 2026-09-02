// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use std::hash::{Hash, Hasher};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Instant;

use arrow_array::{ArrayRef, BooleanArray, Int64Array};
use arrow_schema::DataType;
use datafusion::error::{DataFusionError, Result as DataFusionResult};
use datafusion::logical_expr::{
    ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
};
use datafusion::scalar::ScalarValue;
use moka::future::Cache;
use object_store::path::Path as ObjectPath;
use sha2::{Digest, Sha256};
use tokio::fs::File;
use tokio::io::AsyncReadExt;

use crate::{Error, Result};

const DEFAULT_MAX_BYTES: u64 = 512 * 1024 * 1024;
const DEFAULT_CACHE_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug)]
pub struct SparkBloomFilter {
    num_hash_functions: u32,
    words: Vec<u64>,
    file_bytes: u64,
}

impl SparkBloomFilter {
    fn might_contain_long(&self, item: i64) -> bool {
        let hash1 = murmur3_hash_long(item, 0);
        let hash2 = murmur3_hash_long(item, hash1);
        let bit_size = self.words.len() as i64 * 64;
        for index in 1..=self.num_hash_functions as i32 {
            let mut combined_hash = hash1.wrapping_add(index.wrapping_mul(hash2));
            if combined_hash < 0 {
                combined_hash = !combined_hash;
            }
            let bit_index = (combined_hash as i64 % bit_size) as u64;
            if self.words[(bit_index >> 6) as usize] & (1_u64 << (bit_index & 63)) == 0 {
                return false;
            }
        }
        true
    }
}

#[derive(Debug, Default)]
pub struct ExternalBloomMetrics {
    pub input_rows: AtomicU64,
    pub pass_rows: AtomicU64,
    pub load_nanos: AtomicU64,
    pub bytes: AtomicU64,
}

#[derive(Debug)]
pub struct ExternalBloomContains {
    bloom: Arc<SparkBloomFilter>,
    metrics: Arc<ExternalBloomMetrics>,
    name: String,
    signature: Signature,
}

/// Tests a Spark Bloom V1 sidecar with Spark SQL's `xxhash64(Int64, Int64)` encoding.
#[derive(Debug)]
pub struct ExternalBloomContainsSparkXxHash64I64Pair {
    bloom: Arc<SparkBloomFilter>,
    metrics: Arc<ExternalBloomMetrics>,
    name: String,
    signature: Signature,
}

impl PartialEq for ExternalBloomContainsSparkXxHash64I64Pair {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for ExternalBloomContainsSparkXxHash64I64Pair {}

impl Hash for ExternalBloomContainsSparkXxHash64I64Pair {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl PartialEq for ExternalBloomContains {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name
    }
}

impl Eq for ExternalBloomContains {}

impl Hash for ExternalBloomContains {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.name.hash(state);
    }
}

impl ExternalBloomContains {
    pub(crate) fn new(
        bloom: Arc<SparkBloomFilter>,
        metrics: Arc<ExternalBloomMetrics>,
        sha256: &str,
    ) -> Self {
        Self {
            bloom,
            metrics,
            name: format!("external_bloom_{sha256}"),
            signature: Signature::exact(vec![DataType::Int64], Volatility::Immutable),
        }
    }

    fn evaluate_array(&self, values: &Int64Array) -> BooleanArray {
        self.metrics
            .input_rows
            .fetch_add(values.len() as u64, Ordering::Relaxed);
        let matches = values
            .iter()
            .map(|value| value.is_some_and(|value| self.bloom.might_contain_long(value)))
            .collect::<Vec<_>>();
        self.metrics.pass_rows.fetch_add(
            matches.iter().filter(|value| **value).count() as u64,
            Ordering::Relaxed,
        );
        BooleanArray::from(matches)
    }
}

impl ExternalBloomContainsSparkXxHash64I64Pair {
    pub(crate) fn new(
        bloom: Arc<SparkBloomFilter>,
        metrics: Arc<ExternalBloomMetrics>,
        sha256: &str,
    ) -> Self {
        Self {
            bloom,
            metrics,
            name: format!("external_bloom_spark_xxhash64_i64_pair_{sha256}"),
            signature: Signature::exact(
                vec![DataType::Int64, DataType::Int64],
                Volatility::Immutable,
            ),
        }
    }

    fn evaluate_arrays(&self, first: &Int64Array, second: &Int64Array) -> BooleanArray {
        self.metrics
            .input_rows
            .fetch_add(first.len() as u64, Ordering::Relaxed);
        let matches = first
            .iter()
            .zip(second.iter())
            .map(|(first, second)| match (first, second) {
                (Some(first), Some(second)) => self
                    .bloom
                    .might_contain_long(spark_xxhash64_i64_pair(first, second)),
                _ => false,
            })
            .collect::<Vec<_>>();
        self.metrics.pass_rows.fetch_add(
            matches.iter().filter(|value| **value).count() as u64,
            Ordering::Relaxed,
        );
        BooleanArray::from(matches)
    }
}

impl ScalarUDFImpl for ExternalBloomContains {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DataFusionResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        match &args.args[0] {
            ColumnarValue::Array(array) => {
                let values = array.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "external bloom expected Int64 input, got {}",
                        array.data_type()
                    ))
                })?;
                Ok(ColumnarValue::Array(
                    Arc::new(self.evaluate_array(values)) as ArrayRef
                ))
            }
            ColumnarValue::Scalar(ScalarValue::Int64(value)) => {
                self.metrics.input_rows.fetch_add(1, Ordering::Relaxed);
                let is_match = value.is_some_and(|value| self.bloom.might_contain_long(value));
                self.metrics
                    .pass_rows
                    .fetch_add(u64::from(is_match), Ordering::Relaxed);
                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(is_match))))
            }
            value => Err(DataFusionError::Execution(format!(
                "external bloom expected Int64 input, got {}",
                value.data_type()
            ))),
        }
    }
}

impl ScalarUDFImpl for ExternalBloomContainsSparkXxHash64I64Pair {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, _arg_types: &[DataType]) -> DataFusionResult<DataType> {
        Ok(DataType::Boolean)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> DataFusionResult<ColumnarValue> {
        match (&args.args[0], &args.args[1]) {
            (ColumnarValue::Array(first), ColumnarValue::Array(second)) => {
                let first = first.as_any().downcast_ref::<Int64Array>().ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "external bloom expected first Int64 input, got {}",
                        first.data_type()
                    ))
                })?;
                let second = second
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        DataFusionError::Execution(format!(
                            "external bloom expected second Int64 input, got {}",
                            second.data_type()
                        ))
                    })?;
                if first.len() != second.len() {
                    return Err(DataFusionError::Execution(format!(
                        "external bloom pair inputs have different lengths: {} and {}",
                        first.len(),
                        second.len()
                    )));
                }
                Ok(ColumnarValue::Array(
                    Arc::new(self.evaluate_arrays(first, second)) as ArrayRef,
                ))
            }
            (
                ColumnarValue::Scalar(ScalarValue::Int64(first)),
                ColumnarValue::Scalar(ScalarValue::Int64(second)),
            ) => {
                self.metrics.input_rows.fetch_add(1, Ordering::Relaxed);
                let is_match = match (first, second) {
                    (Some(first), Some(second)) => self
                        .bloom
                        .might_contain_long(spark_xxhash64_i64_pair(*first, *second)),
                    _ => false,
                };
                self.metrics
                    .pass_rows
                    .fetch_add(u64::from(is_match), Ordering::Relaxed);
                Ok(ColumnarValue::Scalar(ScalarValue::Boolean(Some(is_match))))
            }
            (first, second) => Err(DataFusionError::Execution(format!(
                "external bloom expected two Int64 inputs, got {} and {}",
                first.data_type(),
                second.data_type()
            ))),
        }
    }
}

pub struct LoadedBloom {
    pub bloom: Arc<SparkBloomFilter>,
    pub cache_miss: bool,
    pub load_nanos: u64,
}

impl LoadedBloom {
    pub(crate) fn file_bytes(&self) -> u64 {
        self.bloom.file_bytes
    }
}

static BLOOM_CACHE: LazyLock<Cache<(String, String), Arc<SparkBloomFilter>>> =
    LazyLock::new(|| {
        Cache::builder()
            .weigher(|_key, value: &Arc<SparkBloomFilter>| {
                value.file_bytes.saturating_add(64).min(u32::MAX as u64) as u32
            })
            .max_capacity(env_bytes(
                "LANCE_EXTERNAL_BLOOM_CACHE_BYTES",
                DEFAULT_CACHE_BYTES,
            ))
            .build()
    });

pub async fn load(path: &str, sha256: &str) -> Result<LoadedBloom> {
    let cache_miss = Arc::new(AtomicBool::new(false));
    let load_nanos = Arc::new(AtomicU64::new(0));
    let miss_for_loader = cache_miss.clone();
    let nanos_for_loader = load_nanos.clone();
    let path_for_loader = path.to_owned();
    let sha_for_loader = sha256.to_owned();
    let bloom = BLOOM_CACHE
        .try_get_with((path.to_owned(), sha256.to_owned()), async move {
            miss_for_loader.store(true, Ordering::Relaxed);
            let started = Instant::now();
            let result = parse_and_verify(&path_for_loader, &sha_for_loader).await;
            nanos_for_loader.store(
                started.elapsed().as_nanos().min(u64::MAX as u128) as u64,
                Ordering::Relaxed,
            );
            result
        })
        .await
        .map_err(|error: Arc<Error>| Error::cloned(error.to_string()))?;
    Ok(LoadedBloom {
        bloom,
        cache_miss: cache_miss.load(Ordering::Relaxed),
        load_nanos: load_nanos.load(Ordering::Relaxed),
    })
}

async fn parse_and_verify(path: &str, expected_sha256: &str) -> Result<Arc<SparkBloomFilter>> {
    let file = File::open(path).await.map_err(|error| {
        Error::io(format!(
            "failed to open external bloom path '{path}': {error}"
        ))
    })?;
    let file_len = file
        .metadata()
        .await
        .map_err(|error| {
            Error::io(format!(
                "failed to read metadata for external bloom path '{path}': {error}"
            ))
        })?
        .len();
    let max_bytes = env_bytes("LANCE_EXTERNAL_BLOOM_MAX_BYTES", DEFAULT_MAX_BYTES);
    if !(12..=max_bytes).contains(&file_len) {
        return Err(corrupt(
            path,
            format!("invalid file size {file_len}; expected 12..={max_bytes} bytes"),
        ));
    }

    let capacity = usize::try_from(file_len).map_err(|_| {
        corrupt(
            path,
            format!("file size {file_len} cannot be represented on this platform"),
        )
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    file.take(max_bytes.saturating_add(1))
        .read_to_end(&mut bytes)
        .await
        .map_err(|error| {
            Error::io(format!(
                "failed to read external bloom path '{path}': {error}"
            ))
        })?;
    if bytes.len() as u64 != file_len {
        return Err(corrupt(
            path,
            format!(
                "file size changed while reading: metadata={file_len}, actual={}",
                bytes.len()
            ),
        ));
    }

    let actual_sha256 = format!("{:x}", Sha256::digest(&bytes));
    if actual_sha256 != expected_sha256 {
        return Err(corrupt(
            path,
            format!("checksum mismatch: expected {expected_sha256}, got {actual_sha256}"),
        ));
    }

    let version = read_i32(&bytes, 0, path)?;
    let num_hash_functions = read_i32(&bytes, 4, path)?;
    let num_words = read_i32(&bytes, 8, path)?;
    if version != 1 {
        return Err(corrupt(
            path,
            format!("unsupported version {version}; expected 1"),
        ));
    }
    if !(1..=64).contains(&num_hash_functions) {
        return Err(corrupt(
            path,
            format!("invalid num_hash_functions {num_hash_functions}; expected 1..=64"),
        ));
    }
    if num_words < 1 {
        return Err(corrupt(
            path,
            format!("invalid num_words {num_words}; expected at least 1"),
        ));
    }
    let expected_len = 12_u64 + 8_u64 * num_words as u64;
    if file_len != expected_len {
        return Err(corrupt(
            path,
            format!("invalid encoded length {file_len}; header requires {expected_len}"),
        ));
    }

    let mut words = Vec::with_capacity(num_words as usize);
    for chunk in bytes[12..].chunks_exact(8) {
        let word =
            <[u8; 8]>::try_from(chunk).map_err(|_| corrupt(path, "invalid word encoding"))?;
        words.push(u64::from_be_bytes(word));
    }
    Ok(Arc::new(SparkBloomFilter {
        num_hash_functions: num_hash_functions as u32,
        words,
        file_bytes: file_len,
    }))
}

fn read_i32(bytes: &[u8], offset: usize, path: &str) -> Result<i32> {
    let encoded = bytes
        .get(offset..offset + 4)
        .ok_or_else(|| corrupt(path, format!("truncated header at byte {offset}")))?;
    let encoded = <[u8; 4]>::try_from(encoded)
        .map_err(|_| corrupt(path, format!("invalid header at byte {offset}")))?;
    Ok(i32::from_be_bytes(encoded))
}

fn corrupt(path: &str, message: impl Into<String>) -> Error {
    Error::corrupt_file(ObjectPath::from(path.trim_start_matches('/')), message)
}

fn env_bytes(name: &str, default_value: u64) -> u64 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(default_value)
}

fn murmur3_hash_long(input: i64, seed: i32) -> i32 {
    let low = input as i32;
    let high = ((input as u64) >> 32) as i32;
    let hash = mix_h1(seed, mix_k1(low));
    fmix(mix_h1(hash, mix_k1(high)), 8)
}

/// Spark SQL `xxhash64(first, second)` for two non-null Int64 values.
pub(crate) fn spark_xxhash64_i64_pair(first: i64, second: i64) -> i64 {
    xxhash64_long(second, xxhash64_long(first, 42))
}

fn xxhash64_long(input: i64, seed: i64) -> i64 {
    const PRIME1: i64 = 0x9e37_79b1_85eb_ca87_u64 as i64;
    const PRIME2: i64 = 0xc2b2_ae3d_27d4_eb4f_u64 as i64;
    const PRIME3: i64 = 0x1656_67b1_9e37_79f9;
    const PRIME4: i64 = 0x85eb_ca77_c2b2_ae63_u64 as i64;
    const PRIME5: i64 = 0x27d4_eb2f_1656_67c5;

    let mut hash = seed.wrapping_add(PRIME5).wrapping_add(8);
    hash ^= input
        .wrapping_mul(PRIME2)
        .rotate_left(31)
        .wrapping_mul(PRIME1);
    hash = hash
        .rotate_left(27)
        .wrapping_mul(PRIME1)
        .wrapping_add(PRIME4);
    hash ^= (hash as u64 >> 33) as i64;
    hash = hash.wrapping_mul(PRIME2);
    hash ^= (hash as u64 >> 29) as i64;
    hash = hash.wrapping_mul(PRIME3);
    hash ^ (hash as u64 >> 32) as i64
}

fn mix_k1(value: i32) -> i32 {
    value
        .wrapping_mul(0xcc9e_2d51_u32 as i32)
        .rotate_left(15)
        .wrapping_mul(0x1b87_3593)
}

fn mix_h1(hash: i32, value: i32) -> i32 {
    (hash ^ value)
        .rotate_left(13)
        .wrapping_mul(5)
        .wrapping_add(0xe654_6b64_u32 as i32)
}

fn fmix(mut hash: i32, len: i32) -> i32 {
    hash ^= len;
    hash ^= (hash as u32 >> 16) as i32;
    hash = hash.wrapping_mul(0x85eb_ca6b_u32 as i32);
    hash ^= (hash as u32 >> 13) as i32;
    hash = hash.wrapping_mul(0xc2b2_ae35_u32 as i32);
    hash ^ (hash as u32 >> 16) as i32
}

pub fn validate_params(path: &str, column: &str, sha256: &str) -> Result<String> {
    if path.is_empty() || !Path::new(path).is_absolute() {
        return Err(Error::invalid_input(format!(
            "external bloom path must be a non-empty absolute path, got '{path}'"
        )));
    }
    if column.is_empty() || column.contains('.') {
        return Err(Error::invalid_input(format!(
            "external bloom column must be a non-empty top-level name, got '{column}'"
        )));
    }
    if sha256.len() != 64 || !sha256.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(Error::invalid_input(format!(
            "external bloom sha256 must contain exactly 64 hexadecimal characters, got '{sha256}'"
        )));
    }
    Ok(sha256.to_ascii_lowercase())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn test_murmur3_hash_long_matches_spark() {
        assert_eq!(murmur3_hash_long(i64::MIN, 0), 1366273829);
        assert_eq!(murmur3_hash_long(-1, 0), 1651860712);
        assert_eq!(murmur3_hash_long(0, 0), 1669671676);
        assert_eq!(murmur3_hash_long(1, 0), 1392991556);
        assert_eq!(murmur3_hash_long(i64::MAX, 0), -2106506049);
    }

    #[test]
    fn test_xxhash64_i64_pair_matches_spark() {
        assert_eq!(
            spark_xxhash64_i64_pair(i64::MIN, i64::MIN),
            -1467132691780465609
        );
        assert_eq!(spark_xxhash64_i64_pair(-1, 0), 6862433990644673549);
        assert_eq!(spark_xxhash64_i64_pair(0, 0), -9199931545335556226);
        assert_eq!(spark_xxhash64_i64_pair(0, 1), 8281773146650252936);
        assert_eq!(spark_xxhash64_i64_pair(1, -1), -1659187159892528922);
        assert_eq!(
            spark_xxhash64_i64_pair(i64::MAX, i64::MAX),
            3582563888758909168
        );
    }

    #[test]
    fn test_pair_lookup_and_nulls() {
        let bloom = Arc::new(SparkBloomFilter {
            num_hash_functions: 1,
            words: vec![u64::MAX],
            file_bytes: 20,
        });
        let metrics = Arc::new(ExternalBloomMetrics::default());
        let udf =
            ExternalBloomContainsSparkXxHash64I64Pair::new(bloom, metrics.clone(), &"0".repeat(64));
        let result = udf.evaluate_arrays(
            &Int64Array::from(vec![Some(1), None, Some(-1)]),
            &Int64Array::from(vec![Some(2), Some(2), None]),
        );
        assert_eq!(result, BooleanArray::from(vec![true, false, false]));
        assert_eq!(metrics.input_rows.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.pass_rows.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_lookup_and_nulls() {
        let bloom = Arc::new(SparkBloomFilter {
            num_hash_functions: 1,
            words: vec![u64::MAX],
            file_bytes: 20,
        });
        let metrics = Arc::new(ExternalBloomMetrics::default());
        let udf = ExternalBloomContains::new(bloom, metrics.clone(), &"0".repeat(64));
        let result = udf.evaluate_array(&Int64Array::from(vec![Some(1), None, Some(-1)]));
        assert_eq!(result, BooleanArray::from(vec![true, false, true]));
        assert_eq!(metrics.input_rows.load(Ordering::Relaxed), 3);
        assert_eq!(metrics.pass_rows.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_validate_params() {
        assert!(validate_params("relative", "id", &"0".repeat(64)).is_err());
        assert!(validate_params("/tmp/a", "nested.id", &"0".repeat(64)).is_err());
        assert!(validate_params("/tmp/a", "id", "bad").is_err());
        assert_eq!(
            validate_params("/tmp/a", "id", &"A".repeat(64)).unwrap(),
            "a".repeat(64)
        );
    }

    #[tokio::test]
    async fn test_parse_spark_bloom_and_checksum() {
        let item = 0_i64;
        let hash1 = murmur3_hash_long(item, 0);
        let hash2 = murmur3_hash_long(item, hash1);
        let mut combined = hash1.wrapping_add(hash2);
        if combined < 0 {
            combined = !combined;
        }
        let mut word = 0_u64;
        word |= 1_u64 << (combined as u64 % 64);
        let bytes = [
            1_i32.to_be_bytes().as_slice(),
            1_i32.to_be_bytes().as_slice(),
            1_i32.to_be_bytes().as_slice(),
            word.to_be_bytes().as_slice(),
        ]
        .concat();
        let checksum = format!("{:x}", Sha256::digest(&bytes));
        let mut file = NamedTempFile::new().unwrap();
        file.write_all(&bytes).unwrap();
        let path = file.path().to_str().unwrap();

        let bloom = parse_and_verify(path, &checksum).await.unwrap();
        assert!(bloom.might_contain_long(item));

        let first_load = load(path, &checksum).await.unwrap();
        let second_load = load(path, &checksum).await.unwrap();
        assert!(first_load.cache_miss);
        assert!(!second_load.cache_miss);

        let error = parse_and_verify(path, &"0".repeat(64)).await.unwrap_err();
        assert!(matches!(error, Error::CorruptFile { .. }));
        assert!(error.to_string().contains("checksum mismatch"));
    }
}
