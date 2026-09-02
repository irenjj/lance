/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.lance.ipc;

import com.google.common.base.MoreObjects;
import org.apache.arrow.util.Preconditions;

import java.nio.ByteBuffer;
import java.nio.file.Paths;
import java.util.List;
import java.util.Locale;
import java.util.Optional;

/** Lance scan options. */
public class ScanOptions {
  private final Optional<List<Integer>> fragmentIds;
  private final Optional<Long> batchSize;
  private final Optional<List<String>> columns;
  private final Optional<String> filter;
  private final Optional<ByteBuffer> substraitFilter;
  private final Optional<Long> limit;
  private final Optional<Long> offset;
  private final Optional<Query> nearest;
  private final Optional<FullTextQuery> fullTextQuery;
  private final boolean prefilter;
  private final boolean withRowId;
  private final boolean withRowAddress;
  private final int batchReadahead;
  private final Optional<List<ColumnOrdering>> columnOrderings;
  private final boolean useScalarIndex;
  private final Optional<ByteBuffer> substraitAggregate;
  private final boolean collectStats;
  private final boolean fastSearch;
  private final boolean includeDeletedRows;
  private final boolean strictBatchSize;
  private final boolean disableScoringAutoprojection;
  private final Optional<ExternalBloomOptions> externalBloom;

  public ScanOptions(
      Optional<List<Integer>> fragmentIds,
      Optional<Long> batchSize,
      Optional<List<String>> columns,
      Optional<String> filter,
      Optional<ByteBuffer> substraitFilter,
      Optional<Long> limit,
      Optional<Long> offset,
      Optional<Query> nearest,
      Optional<FullTextQuery> fullTextQuery,
      boolean prefilter,
      boolean withRowId,
      boolean withRowAddress,
      int batchReadahead,
      Optional<List<ColumnOrdering>> columnOrderings,
      boolean useScalarIndex,
      Optional<ByteBuffer> substraitAggregate,
      boolean collectStats) {
    this(
        fragmentIds,
        batchSize,
        columns,
        filter,
        substraitFilter,
        limit,
        offset,
        nearest,
        fullTextQuery,
        prefilter,
        withRowId,
        withRowAddress,
        batchReadahead,
        columnOrderings,
        useScalarIndex,
        substraitAggregate,
        collectStats,
        false,
        false,
        false,
        false);
  }

  /**
   * Constructor for LanceScanOptions.
   *
   * @param fragmentIds the id of the fragments to scan
   * @param batchSize Maximum row number of each returned ArrowRecordBatch. Optional, use
   *     Optional.empty() if unspecified.
   * @param columns (Optional) Projected columns. Optional.empty() for scanning all columns.
   *     Otherwise, only columns present in the List will be scanned.
   * @param filter (Optional) Filter expression. Optional.empty() for no filter.
   * @param substraitFilter (Optional) Substrait filter expression.
   * @param filter (Optional) Filter expression. Optional.empty() for no filter.
   * @param limit (Optional) Maximum number of rows to return.
   * @param offset (Optional) Number of rows to skip before returning results.
   * @param withRowId Whether to include the row ID in the results.
   * @param withRowAddress Whether to include the row address in the results.
   * @param nearest (Optional) Nearest neighbor query.
   * @param batchReadahead Number of batches to read ahead.
   * @param columnOrderings (Optional) Column orderings for result sorting.
   * @param useScalarIndex Whether to use scalar indices for the scan. Default is true.
   * @param substraitAggregate (Optional) Substrait aggregate expression for aggregate pushdown.
   * @param collectStats Whether to collect scan execution statistics. Default is false.
   * @param fastSearch Whether to only search indexed fragments. Default is false.
   */
  public ScanOptions(
      Optional<List<Integer>> fragmentIds,
      Optional<Long> batchSize,
      Optional<List<String>> columns,
      Optional<String> filter,
      Optional<ByteBuffer> substraitFilter,
      Optional<Long> limit,
      Optional<Long> offset,
      Optional<Query> nearest,
      Optional<FullTextQuery> fullTextQuery,
      boolean prefilter,
      boolean withRowId,
      boolean withRowAddress,
      int batchReadahead,
      Optional<List<ColumnOrdering>> columnOrderings,
      boolean useScalarIndex,
      Optional<ByteBuffer> substraitAggregate,
      boolean collectStats,
      boolean fastSearch,
      boolean includeDeletedRows,
      boolean strictBatchSize,
      boolean disableScoringAutoprojection) {
    this(
        fragmentIds,
        batchSize,
        columns,
        filter,
        substraitFilter,
        limit,
        offset,
        nearest,
        fullTextQuery,
        prefilter,
        withRowId,
        withRowAddress,
        batchReadahead,
        columnOrderings,
        useScalarIndex,
        substraitAggregate,
        collectStats,
        fastSearch,
        includeDeletedRows,
        strictBatchSize,
        disableScoringAutoprojection,
        Optional.empty());
  }

  private ScanOptions(
      Optional<List<Integer>> fragmentIds,
      Optional<Long> batchSize,
      Optional<List<String>> columns,
      Optional<String> filter,
      Optional<ByteBuffer> substraitFilter,
      Optional<Long> limit,
      Optional<Long> offset,
      Optional<Query> nearest,
      Optional<FullTextQuery> fullTextQuery,
      boolean prefilter,
      boolean withRowId,
      boolean withRowAddress,
      int batchReadahead,
      Optional<List<ColumnOrdering>> columnOrderings,
      boolean useScalarIndex,
      Optional<ByteBuffer> substraitAggregate,
      boolean collectStats,
      boolean fastSearch,
      boolean includeDeletedRows,
      boolean strictBatchSize,
      boolean disableScoringAutoprojection,
      Optional<ExternalBloomOptions> externalBloom) {
    Preconditions.checkArgument(
        !(filter.isPresent() && substraitFilter.isPresent()),
        "cannot set both substrait filter and string filter");
    Preconditions.checkArgument(
        batchReadahead > 0, "batchReadahead must be greater than 0, got %s", batchReadahead);
    this.fragmentIds = fragmentIds;
    this.batchSize = batchSize;
    this.columns = columns;
    this.filter = filter;
    this.substraitFilter = substraitFilter;
    this.limit = limit;
    this.offset = offset;
    this.nearest = nearest;
    this.fullTextQuery = fullTextQuery;
    this.prefilter = prefilter;
    this.withRowId = withRowId;
    this.withRowAddress = withRowAddress;
    this.batchReadahead = batchReadahead;
    this.columnOrderings = columnOrderings;
    this.useScalarIndex = useScalarIndex;
    this.substraitAggregate = substraitAggregate;
    this.collectStats = collectStats;
    this.fastSearch = fastSearch;
    this.includeDeletedRows = includeDeletedRows;
    this.strictBatchSize = strictBatchSize;
    this.disableScoringAutoprojection = disableScoringAutoprojection;
    this.externalBloom = externalBloom;
  }

  /**
   * Get the fragment ids.
   *
   * @return Optional containing the fragment ids if specified, otherwise empty.
   */
  public Optional<List<Integer>> getFragmentIds() {
    return fragmentIds;
  }

  /**
   * Get the batch size.
   *
   * @return Optional containing the batch size if specified, otherwise empty.
   */
  public Optional<Long> getBatchSize() {
    return batchSize;
  }

  /**
   * Get the columns.
   *
   * @return Optional containing the columns if specified, otherwise empty.
   */
  public Optional<List<String>> getColumns() {
    return columns;
  }

  /**
   * Get the filter.
   *
   * @return Optional containing the filter if specified, otherwise empty.
   */
  public Optional<String> getFilter() {
    return filter;
  }

  /**
   * Get the substrait filter.
   *
   * @return Optional containing the substrait filter if specified, otherwise empty.
   */
  public Optional<ByteBuffer> getSubstraitFilter() {
    return substraitFilter;
  }

  /**
   * Get the limit.
   *
   * @return Optional containing the limit if specified, otherwise empty.
   */
  public Optional<Long> getLimit() {
    return limit;
  }

  /**
   * Get the offset.
   *
   * @return Optional containing the offset if specified, otherwise empty.
   */
  public Optional<Long> getOffset() {
    return offset;
  }

  /**
   * Get the nearest neighbor query.
   *
   * @return Optional containing the nearest neighbor query if specified, otherwise empty.
   */
  public Optional<Query> getNearest() {
    return nearest;
  }

  /**
   * Get the full text search query.
   *
   * @return Optional containing the full text search query if specified, otherwise empty.
   */
  public Optional<FullTextQuery> getFullTextQuery() {
    return fullTextQuery;
  }

  /**
   * Get whether to prefilter before nearest neighbor search.
   *
   * @return true if prefilter should be applied, false otherwise.
   */
  public boolean isPrefilter() {
    return prefilter;
  }

  /**
   * Get whether to include the row ID.
   *
   * @return true if row ID should be included, false otherwise.
   */
  public boolean isWithRowId() {
    return withRowId;
  }

  /**
   * Get whether to include the row address.
   *
   * @return true if row address should be included, false otherwise.
   */
  public boolean isWithRowAddress() {
    return withRowAddress;
  }

  /**
   * Get the batch readahead.
   *
   * @return the number of batches to read ahead.
   */
  public int getBatchReadahead() {
    return batchReadahead;
  }

  public Optional<List<ColumnOrdering>> getColumnOrderings() {
    return columnOrderings;
  }

  /**
   * Get whether to use scalar indices for the scan.
   *
   * @return true if scalar indices should be used, false otherwise.
   */
  public boolean isUseScalarIndex() {
    return useScalarIndex;
  }

  /**
   * Get whether to only search indexed fragments.
   *
   * @return true if unindexed fragments should be skipped, false otherwise.
   */
  public boolean isFastSearch() {
    return fastSearch;
  }

  /**
   * Get the substrait aggregate expression.
   *
   * @return Optional containing the substrait aggregate if specified, otherwise empty.
   */
  public Optional<ByteBuffer> getSubstraitAggregate() {
    return substraitAggregate;
  }

  public boolean isCollectStats() {
    return collectStats;
  }

  /**
   * Get whether to include deleted rows in scan results.
   *
   * @return true if deleted rows should be included, false otherwise.
   */
  public boolean isIncludeDeletedRows() {
    return includeDeletedRows;
  }

  /**
   * Get whether to enforce strict batch sizing.
   *
   * @return true if batch sizes must be strictly enforced, false otherwise.
   */
  public boolean isStrictBatchSize() {
    return strictBatchSize;
  }

  /**
   * Get whether to disable scoring autoprojection.
   *
   * @return true if scoring column autoprojection is disabled, false otherwise.
   */
  public boolean isDisableScoringAutoprojection() {
    return disableScoringAutoprojection;
  }

  /**
   * Get the external Bloom filter options.
   *
   * @return external Bloom options when configured, otherwise empty
   */
  public Optional<ExternalBloomOptions> getExternalBloom() {
    return externalBloom;
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("fragmentIds", fragmentIds.orElse(null))
        .add("batchSize", batchSize.orElse(null))
        .add("columns", columns.orElse(null))
        .add("filter", filter.orElse(null))
        .add(
            "substraitFilter",
            substraitFilter.map(buf -> "ByteBuffer[" + buf.remaining() + " bytes]").orElse(null))
        .add("limit", limit.orElse(null))
        .add("offset", offset.orElse(null))
        .add("nearest", nearest.orElse(null))
        .add("fullTextQuery", fullTextQuery.orElse(null))
        .add("prefilter", prefilter)
        .add("withRowId", withRowId)
        .add("WithRowAddress", withRowAddress)
        .add("batchReadahead", batchReadahead)
        .add("columnOrdering", columnOrderings)
        .add("useScalarIndex", useScalarIndex)
        .add("fastSearch", fastSearch)
        .add(
            "substraitAggregate",
            substraitAggregate.map(buf -> "ByteBuffer[" + buf.remaining() + " bytes]").orElse(null))
        .add("collectStats", collectStats)
        .add("includeDeletedRows", includeDeletedRows)
        .add("strictBatchSize", strictBatchSize)
        .add("disableScoringAutoprojection", disableScoringAutoprojection)
        .add("externalBloom", externalBloom.orElse(null))
        .toString();
  }

  /** Options for filtering an Int64 column with a Spark Bloom V1 sidecar. */
  public static final class ExternalBloomOptions {
    public static final String INT64_V1 = "int64_v1";
    public static final String SPARK_XXHASH64_I64_PAIR_V1 = "spark_xxhash64_i64_pair_v1";

    private final String path;
    private final String column;
    private final Optional<String> secondColumn;
    private final String keyEncoding;
    private final String sha256;

    private ExternalBloomOptions(String path, String column, String sha256) {
      this(path, column, Optional.empty(), INT64_V1, sha256);
    }

    private ExternalBloomOptions(
        String path,
        String column,
        Optional<String> secondColumn,
        String keyEncoding,
        String sha256) {
      Preconditions.checkNotNull(path, "external bloom path must not be null");
      Preconditions.checkNotNull(column, "external bloom column must not be null");
      Preconditions.checkNotNull(secondColumn, "external bloom second column must not be null");
      Preconditions.checkNotNull(keyEncoding, "external bloom key encoding must not be null");
      Preconditions.checkNotNull(sha256, "external bloom sha256 must not be null");
      Preconditions.checkArgument(
          !path.isEmpty() && Paths.get(path).isAbsolute(),
          "external bloom path must be a non-empty absolute path, got %s",
          path);
      Preconditions.checkArgument(
          !column.isEmpty() && !column.contains("."),
          "external bloom column must be a non-empty top-level name, got %s",
          column);
      secondColumn.ifPresent(
          value -> {
            Preconditions.checkArgument(
                !value.isEmpty() && !value.contains("."),
                "external bloom second column must be a non-empty top-level name, got %s",
                value);
            Preconditions.checkArgument(
                !column.equals(value), "external bloom pair columns must be distinct");
          });
      Preconditions.checkArgument(
          (INT64_V1.equals(keyEncoding) && secondColumn.isEmpty())
              || (SPARK_XXHASH64_I64_PAIR_V1.equals(keyEncoding) && secondColumn.isPresent()),
          "unsupported external bloom key encoding/column combination: %s",
          keyEncoding);
      Preconditions.checkArgument(
          sha256.matches("[0-9a-fA-F]{64}"),
          "external bloom sha256 must contain exactly 64 hexadecimal characters, got %s",
          sha256);
      this.path = path;
      this.column = column;
      this.secondColumn = secondColumn;
      this.keyEncoding = keyEncoding;
      this.sha256 = sha256.toLowerCase(Locale.ROOT);
    }

    /**
     * @return absolute local or NFS sidecar path
     */
    public String getPath() {
      return path;
    }

    /**
     * @return top-level Int64 column to filter
     */
    public String getColumn() {
      return column;
    }

    /**
     * @return second top-level Int64 column for pair encoding, or empty for single-column mode
     */
    public Optional<String> getSecondColumn() {
      return secondColumn;
    }

    /**
     * @return versioned key encoding contract
     */
    public String getKeyEncoding() {
      return keyEncoding;
    }

    /**
     * @return lowercase SHA-256 of the sidecar
     */
    public String getSha256() {
      return sha256;
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("path", path)
          .add("column", column)
          .add("secondColumn", secondColumn.orElse(null))
          .add("keyEncoding", keyEncoding)
          .add("sha256", sha256)
          .toString();
    }
  }

  /** Builder for constructing LanceScanOptions. */
  public static class Builder {
    private Optional<List<Integer>> fragmentIds = Optional.empty();
    private Optional<Long> batchSize = Optional.empty();
    private Optional<List<String>> columns = Optional.empty();
    private Optional<String> filter = Optional.empty();
    private Optional<ByteBuffer> substraitFilter = Optional.empty();
    private Optional<Long> limit = Optional.empty();
    private Optional<Long> offset = Optional.empty();
    private Optional<Query> nearest = Optional.empty();
    private Optional<FullTextQuery> fullTextQuery = Optional.empty();
    private boolean prefilter = false;
    private boolean withRowId = false;
    private boolean withRowAddress = false;
    private int batchReadahead = 16;
    private Optional<List<ColumnOrdering>> columnOrderings = Optional.empty();
    private boolean useScalarIndex = true;
    private boolean fastSearch = false;
    private Optional<ByteBuffer> substraitAggregate = Optional.empty();
    private boolean collectStats = false;
    private boolean includeDeletedRows = false;
    private boolean strictBatchSize = false;
    private boolean disableScoringAutoprojection = false;
    private Optional<ExternalBloomOptions> externalBloom = Optional.empty();

    public Builder() {}

    /**
     * Create a builder from another scan options.
     *
     * @param options another scan options
     */
    public Builder(ScanOptions options) {
      this.fragmentIds = options.getFragmentIds();
      this.batchSize = options.getBatchSize();
      this.columns = options.getColumns();
      this.filter = options.getFilter();
      this.substraitFilter = options.getSubstraitFilter();
      this.limit = options.getLimit();
      this.offset = options.getOffset();
      this.nearest = options.getNearest();
      this.fullTextQuery = options.getFullTextQuery();
      this.prefilter = options.isPrefilter();
      this.withRowId = options.isWithRowId();
      this.withRowAddress = options.isWithRowAddress();
      this.batchReadahead = options.getBatchReadahead();
      this.columnOrderings = options.getColumnOrderings();
      this.useScalarIndex = options.isUseScalarIndex();
      this.fastSearch = options.isFastSearch();
      this.substraitAggregate = options.getSubstraitAggregate();
      this.collectStats = options.isCollectStats();
      this.includeDeletedRows = options.isIncludeDeletedRows();
      this.strictBatchSize = options.isStrictBatchSize();
      this.disableScoringAutoprojection = options.isDisableScoringAutoprojection();
      this.externalBloom = options.getExternalBloom();
    }

    /**
     * Set the fragment ids.
     *
     * @param fragmentIds the id of the fragments to scan
     * @return Builder instance for method chaining.
     */
    public Builder fragmentIds(List<Integer> fragmentIds) {
      this.fragmentIds = Optional.of(fragmentIds);
      return this;
    }

    /**
     * Set the batch size.
     *
     * @param batchSize Maximum row number of each returned ArrowRecordBatch.
     * @return Builder instance for method chaining.
     */
    public Builder batchSize(long batchSize) {
      this.batchSize = Optional.of(batchSize);
      return this;
    }

    /**
     * Set the columns.
     *
     * @param columns Projected columns.
     * @return Builder instance for method chaining.
     */
    public Builder columns(List<String> columns) {
      this.columns = Optional.of(columns);
      return this;
    }

    /**
     * Set the filter.
     *
     * @param filter Filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder filter(String filter) {
      this.filter = Optional.of(filter);
      return this;
    }

    /**
     * Set the substrait filter.
     *
     * @param substraitFilter Substrait filter expression.
     * @return Builder instance for method chaining.
     */
    public Builder substraitFilter(ByteBuffer substraitFilter) {
      this.substraitFilter = Optional.of(substraitFilter);
      return this;
    }

    /**
     * Set the limit.
     *
     * @param limit Maximum number of rows to return.
     * @return Builder instance for method chaining.
     */
    public Builder limit(long limit) {
      this.limit = Optional.of(limit);
      return this;
    }

    /**
     * Set the offset.
     *
     * @param offset Number of rows to skip before returning results.
     * @return Builder instance for method chaining.
     */
    public Builder offset(long offset) {
      this.offset = Optional.of(offset);
      return this;
    }

    /**
     * Set the nearest neighbor query.
     *
     * @param nearest The nearest neighbor query.
     * @return Builder instance for method chaining.
     */
    public Builder nearest(Query nearest) {
      this.nearest = Optional.of(nearest);
      return this;
    }

    /**
     * Set the full text search query.
     *
     * @param fullTextQuery full text search query definition.
     * @return Builder instance for method chaining.
     */
    public Builder fullTextQuery(FullTextQuery fullTextQuery) {
      this.fullTextQuery = Optional.ofNullable(fullTextQuery);
      return this;
    }

    /**
     * Set whether to prefilter during nearest neighbor search.
     *
     * @param prefilter true to apply prefilter, false otherwise.
     * @return Builder instance for method chaining.
     */
    public Builder prefilter(boolean prefilter) {
      this.prefilter = prefilter;
      return this;
    }

    /**
     * Set whether to include the row ID.
     *
     * @param withRowId true to include row ID, false otherwise.
     * @return Builder instance for method chaining.
     */
    public Builder withRowId(boolean withRowId) {
      this.withRowId = withRowId;
      return this;
    }

    /**
     * Set whether to include the row addr.
     *
     * @param withRowAddress true to include row ID, false otherwise.
     * @return Builder instance for method chaining.
     */
    public Builder withRowAddress(boolean withRowAddress) {
      this.withRowAddress = withRowAddress;
      return this;
    }

    /**
     * Set the batch readahead.
     *
     * @param batchReadahead Number of batches to read ahead.
     * @return Builder instance for method chaining.
     */
    public Builder batchReadahead(int batchReadahead) {
      this.batchReadahead = batchReadahead;
      return this;
    }

    public Builder setColumnOrderings(List<ColumnOrdering> columnOrderings) {
      this.columnOrderings = Optional.of(columnOrderings);
      return this;
    }

    /**
     * Set whether to use scalar indices for the scan.
     *
     * <p>Scans will use scalar indices, when available, to optimize queries with filters. However,
     * in some corner cases, scalar indices may make performance worse. This parameter allows users
     * to disable scalar indices in these cases.
     *
     * @param useScalarIndex true to use scalar indices, false otherwise. Default is true.
     * @return Builder instance for method chaining.
     */
    public Builder useScalarIndex(boolean useScalarIndex) {
      this.useScalarIndex = useScalarIndex;
      return this;
    }

    /**
     * Set whether to only search indexed fragments.
     *
     * <p>This is a weak-consistency mode for vector search, full text search, and scalar-indexed
     * filters. It can reduce latency by skipping recently appended fragments that are not covered
     * by the relevant index.
     *
     * @param fastSearch true to skip unindexed fragments, false otherwise. Default is false.
     * @return Builder instance for method chaining.
     */
    public Builder fastSearch(boolean fastSearch) {
      this.fastSearch = fastSearch;
      return this;
    }

    /**
     * Set the substrait aggregate expression.
     *
     * @param substraitAggregate Substrait aggregate expression.
     * @return Builder instance for method chaining.
     */
    public Builder substraitAggregate(ByteBuffer substraitAggregate) {
      this.substraitAggregate = Optional.of(substraitAggregate);
      return this;
    }

    /**
     * Enable or disable scan execution statistics collection.
     *
     * <p>When enabled, the native scanner will collect statistics (see {@link ScanStats}) for the
     * scan and make them available via {@link LanceScanner#getStats()} after the scan stream is
     * fully consumed and closed.
     *
     * <p>Default is false.
     */
    public Builder collectStats(boolean collectStats) {
      this.collectStats = collectStats;
      return this;
    }

    /**
     * Set whether to include deleted rows in scan results. Default is false.
     *
     * @param includeDeletedRows whether to include deleted rows
     * @return Builder instance for method chaining.
     */
    public Builder includeDeletedRows(boolean includeDeletedRows) {
      this.includeDeletedRows = includeDeletedRows;
      return this;
    }

    /**
     * Set whether to enforce strict batch sizing. Default is false.
     *
     * @param strictBatchSize whether to enforce strict batch sizing
     * @return Builder instance for method chaining.
     */
    public Builder strictBatchSize(boolean strictBatchSize) {
      this.strictBatchSize = strictBatchSize;
      return this;
    }

    /**
     * Set whether to disable scoring column autoprojection. Default is false.
     *
     * @param disableScoringAutoprojection whether to disable autoprojection
     * @return Builder instance for method chaining.
     */
    public Builder disableScoringAutoprojection(boolean disableScoringAutoprojection) {
      this.disableScoringAutoprojection = disableScoringAutoprojection;
      return this;
    }

    /**
     * Filter an Int64 column with a Spark Bloom V1 sidecar.
     *
     * @param path absolute local or NFS sidecar path
     * @param column top-level Int64 column to filter
     * @param sha256 SHA-256 of the sidecar
     * @return Builder instance for method chaining
     */
    public Builder externalBloom(String path, String column, String sha256) {
      this.externalBloom = Optional.of(new ExternalBloomOptions(path, column, sha256));
      return this;
    }

    /**
     * Filter Spark SQL {@code xxhash64(firstColumn, secondColumn)} with a Spark Bloom V1 sidecar.
     *
     * @param path absolute local or NFS sidecar path
     * @param firstColumn first top-level Int64 column
     * @param secondColumn second top-level Int64 column
     * @param sha256 SHA-256 of the sidecar
     * @return Builder instance for method chaining
     */
    public Builder externalBloomSparkXxHash64I64Pair(
        String path, String firstColumn, String secondColumn, String sha256) {
      this.externalBloom =
          Optional.of(
              new ExternalBloomOptions(
                  path,
                  firstColumn,
                  Optional.of(secondColumn),
                  ExternalBloomOptions.SPARK_XXHASH64_I64_PAIR_V1,
                  sha256));
      return this;
    }

    /**
     * Build the LanceScanOptions instance.
     *
     * @return LanceScanOptions instance with the specified parameters.
     */
    public ScanOptions build() {
      return new ScanOptions(
          fragmentIds,
          batchSize,
          columns,
          filter,
          substraitFilter,
          limit,
          offset,
          nearest,
          fullTextQuery,
          prefilter,
          withRowId,
          withRowAddress,
          batchReadahead,
          columnOrderings,
          useScalarIndex,
          substraitAggregate,
          collectStats,
          fastSearch,
          includeDeletedRows,
          strictBatchSize,
          disableScoringAutoprojection,
          externalBloom);
    }
  }
}
