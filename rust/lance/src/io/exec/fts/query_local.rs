// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{Array, RecordBatch, UInt64Array};
use lance_core::{Error, Result, datatypes::Schema};
use lance_index::scalar::InvertedIndexParams;
use lance_index::scalar::inverted::query::{
    FtsQuery, MatchQuery, Operator, Tokens, try_collect_query_tokens,
};
use lance_index::scalar::inverted::tokenizer::document_tokenizer::LanceTokenizer;
use lance_index::scalar::inverted::{MemBM25Scorer, Scorer};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::index::scalar::inverted::{ResolvedFtsField, resolve_fts_field};

/// Term-only postings owned by one residual scan shard.
#[derive(Debug)]
pub(super) struct QueryLocalFtsIndex {
    column: String,
    params: InvertedIndexParams,
    tokenizer: Box<dyn LanceTokenizer>,
    field: Option<ResolvedFtsField>,
    documents: Vec<(u64, u32)>,
    postings: FxHashMap<String, Vec<Posting>>,
}

#[derive(Debug)]
struct Posting {
    document: usize,
    positions: Vec<u32>,
}

type LeafResults = Vec<Vec<(u64, f32)>>;

enum LeafKind {
    Match(Operator),
    Phrase(u32),
}

struct ExactLeaf<'a> {
    text: &'a str,
    kind: LeafKind,
}

#[derive(Default)]
struct GroupMatch {
    score: f32,
    positions: Vec<u32>,
}

impl QueryLocalFtsIndex {
    pub(super) fn try_with_loaded_tokenizer(
        column: String,
        params: InvertedIndexParams,
        tokenizer: Box<dyn LanceTokenizer>,
    ) -> Result<Self> {
        params.validate_format_version()?;
        Ok(Self {
            column,
            params,
            tokenizer,
            field: None,
            documents: Vec::new(),
            postings: FxHashMap::default(),
        })
    }

    pub(super) fn empty_sibling(&self) -> Self {
        Self {
            column: self.column.clone(),
            params: self.params.clone(),
            tokenizer: self.tokenizer.clone(),
            field: self.field.clone(),
            documents: Vec::new(),
            postings: FxHashMap::default(),
        }
    }

    pub(super) fn exact_query_terms(&self, query: &FtsQuery) -> Result<Vec<String>> {
        let mut tokenizer = self.tokenizer.clone();
        let mut seen = FxHashSet::default();
        let mut terms = Vec::new();
        for leaf in exact_leaves(query)? {
            let tokens = try_collect_query_tokens(leaf.text, &mut tokenizer)?;
            for index in 0..tokens.len() {
                let term = tokens.get_token(index);
                if seen.insert(term.to_owned()) {
                    terms.push(term.to_owned());
                }
            }
        }
        Ok(terms)
    }

    pub(super) fn insert_with_row_ids_for_terms(
        &mut self,
        batch: &RecordBatch,
        row_ids: &UInt64Array,
        terms: &FxHashSet<String>,
    ) -> Result<QueryLocalFtsStats> {
        if row_ids.len() != batch.num_rows() || row_ids.null_count() != 0 {
            return Err(Error::invalid_input(format!(
                "Residual FTS needs {} non-null row IDs, got len={} nulls={}",
                batch.num_rows(),
                row_ids.len(),
                row_ids.null_count()
            )));
        }
        let field = match &self.field {
            Some(field) => field,
            None => {
                let schema = Schema::try_from(batch.schema().as_ref())?;
                self.field.insert(resolve_fts_field(
                    &schema,
                    &self.column,
                    self.params.get_document_granularity(),
                )?)
            }
        };
        let mut stats = QueryLocalFtsStats::default();
        let discount_overlaps = self.tokenizer.discount_overlaps();
        let keep_empty = self.params.get_document_granularity().is_list_element();
        // Count every document and all of its tokens for BM25, even if it has no retained terms.
        for document in field.documents_from_batch(batch)? {
            let mut stream = self.tokenizer.token_stream_for_doc(&document.text);
            let mut length = 0u32;
            let mut previous_position = None;
            let document_id = self.documents.len();
            let mut has_postings = false;
            while let Some(token) = stream.try_next().map_err(Error::invalid_input)? {
                let position = u32::try_from(token.position).map_err(|_| {
                    Error::invalid_input(format!(
                        "Residual FTS token position {} overflows",
                        token.position
                    ))
                })?;
                if !discount_overlaps || previous_position != Some(position) {
                    length = length.checked_add(1).ok_or_else(|| {
                        Error::invalid_input("Residual FTS document length overflows")
                    })?;
                }
                previous_position = Some(position);
                if terms.contains(token.text.as_str()) {
                    has_postings = true;
                    let postings =
                        if let Some(postings) = self.postings.get_mut(token.text.as_str()) {
                            postings
                        } else {
                            self.postings.entry(token.text.clone()).or_default()
                        };
                    if let Some(posting) = postings.last_mut().filter(|p| p.document == document_id)
                    {
                        posting.positions.push(position);
                    } else {
                        postings.push(Posting {
                            document: document_id,
                            positions: vec![position],
                        });
                        *stats.token_docs.entry(token.text.clone()).or_default() += 1;
                    }
                }
            }
            if keep_empty || length > 0 {
                stats.doc_count += 1;
                stats.total_tokens = stats
                    .total_tokens
                    .checked_add(u64::from(length))
                    .ok_or_else(|| Error::internal("query-local FTS total token count overflow"))?;
            }
            if has_postings {
                self.documents
                    .push((row_ids.value(document.row_index), length));
            }
        }
        Ok(stats)
    }

    pub(super) fn exact_leaf_results(
        self,
        query: &FtsQuery,
        scorer: &MemBM25Scorer,
    ) -> Result<LeafResults> {
        let mut tokenizer = self.tokenizer.clone();
        exact_leaves(query)?
            .into_iter()
            .map(|leaf| {
                let tokens = try_collect_query_tokens(leaf.text, &mut tokenizer)?;
                Ok(self.score_leaf(&tokens, leaf.kind, scorer))
            })
            .collect()
    }

    fn score_leaf(
        &self,
        tokens: &Tokens,
        kind: LeafKind,
        scorer: &MemBM25Scorer,
    ) -> Vec<(u64, f32)> {
        if tokens.is_empty() || scorer.num_docs() == 0 {
            return Vec::new();
        }
        let mut groups: Vec<(u32, Vec<&str>)> = Vec::with_capacity(tokens.len());
        for index in 0..tokens.len() {
            let position = match kind {
                LeafKind::Phrase(_) => tokens.position(index),
                LeafKind::Match(Operator::And) => tokens.term_group(index),
                LeafKind::Match(Operator::Or) => 0,
            };
            let token = tokens.get_token(index);
            match groups.last_mut() {
                Some((previous, terms)) if *previous == position => {
                    if matches!(kind, LeafKind::Match(Operator::Or)) || !terms.contains(&token) {
                        terms.push(token);
                    }
                }
                _ => groups.push((position, vec![token])),
            }
        }
        let phrase_slop = match kind {
            LeafKind::Phrase(slop) if groups.len() > 1 => {
                if !self.params.has_positions() {
                    return Vec::new();
                }
                Some(slop)
            }
            _ => None,
        };
        let group_matches: Vec<_> = groups
            .iter()
            .map(|(_, terms)| self.score_group(terms, scorer, phrase_slop.is_some()))
            .collect();
        let mut results = Vec::new();
        for document in group_matches[0].keys() {
            let Some(matches) = group_matches
                .iter()
                .map(|group| group.get(document))
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if let Some(slop) = phrase_slop
                && !phrase_matches(&groups, &matches, slop)
            {
                continue;
            }
            results.push((
                self.documents[*document].0,
                matches.iter().map(|entry| entry.score).sum(),
            ));
        }
        results
    }

    fn score_group(
        &self,
        terms: &[&str],
        scorer: &MemBM25Scorer,
        with_positions: bool,
    ) -> FxHashMap<usize, GroupMatch> {
        let mut matches = FxHashMap::<usize, GroupMatch>::default();
        for term in terms {
            let Some(postings) = self.postings.get(*term) else {
                continue;
            };
            let weight = scorer.query_weight(term);
            for posting in postings {
                let entry = matches.entry(posting.document).or_default();
                if weight != 0.0 {
                    let length = self.documents[posting.document].1;
                    entry.score +=
                        weight * scorer.doc_weight(posting.positions.len() as u32, length);
                }
                if with_positions {
                    entry.positions.extend(&posting.positions);
                }
            }
        }
        if with_positions {
            for entry in matches.values_mut() {
                entry.positions.sort_unstable();
            }
        }
        matches
    }
}

fn phrase_matches(groups: &[(u32, Vec<&str>)], matches: &[&GroupMatch], slop: u32) -> bool {
    matches[0].positions.iter().any(|&first| {
        let mut previous = u64::from(first);
        // Filtering removes tokens without collapsing their positions.
        for (pair, entry) in groups.windows(2).zip(&matches[1..]) {
            let Some(delta) = pair[1].0.checked_sub(pair[0].0) else {
                return false;
            };
            let minimum = previous + u64::from(delta);
            let next = entry
                .positions
                .partition_point(|&position| u64::from(position) < minimum);
            let Some(&position) = entry.positions.get(next) else {
                return false;
            };
            previous = u64::from(position);
            if previous > minimum + u64::from(slop) {
                return false;
            }
        }
        true
    })
}

fn exact_leaves(query: &FtsQuery) -> Result<Vec<ExactLeaf<'_>>> {
    fn add_match<'a>(query: &'a MatchQuery, leaves: &mut Vec<ExactLeaf<'a>>) -> Result<()> {
        if query.fuzziness != Some(0) {
            return Err(Error::invalid_input(
                "residual compound FTS only supports exact Match leaves",
            ));
        }
        leaves.push(ExactLeaf {
            text: &query.terms,
            kind: LeafKind::Match(query.operator),
        });
        Ok(())
    }

    fn visit<'a>(query: &'a FtsQuery, leaves: &mut Vec<ExactLeaf<'a>>) -> Result<()> {
        match query {
            FtsQuery::Match(query) => add_match(query, leaves)?,
            FtsQuery::Phrase(query) => leaves.push(ExactLeaf {
                text: &query.terms,
                kind: LeafKind::Phrase(query.slop),
            }),
            FtsQuery::Boost(query) => {
                visit(&query.positive, leaves)?;
                visit(&query.negative, leaves)?;
            }
            FtsQuery::MultiMatch(query) => {
                for query in &query.match_queries {
                    add_match(query, leaves)?;
                }
            }
            FtsQuery::Boolean(query) => {
                for query in query
                    .should
                    .iter()
                    .chain(&query.must)
                    .chain(&query.must_not)
                {
                    visit(query, leaves)?;
                }
            }
            FtsQuery::CombinedFields(_) => {
                return Err(Error::invalid_input(
                    "residual compound FTS does not support combined_fields (BM25F) leaves",
                ));
            }
        }
        Ok(())
    }

    let mut leaves = Vec::new();
    visit(query, &mut leaves)?;
    Ok(leaves)
}

#[derive(Debug, Default)]
pub(super) struct QueryLocalFtsStats {
    doc_count: usize,
    total_tokens: u64,
    token_docs: FxHashMap<String, usize>,
}

impl QueryLocalFtsStats {
    pub(super) fn checked_add_assign(&mut self, other: Self) -> Result<()> {
        self.doc_count = self
            .doc_count
            .checked_add(other.doc_count)
            .ok_or_else(|| Error::internal("query-local FTS document count overflow"))?;
        self.total_tokens = self
            .total_tokens
            .checked_add(other.total_tokens)
            .ok_or_else(|| Error::internal("query-local FTS total token count overflow"))?;
        for (token, df) in other.token_docs {
            let current = self.token_docs.entry(token).or_default();
            *current = current
                .checked_add(df)
                .ok_or_else(|| Error::internal("query-local FTS term document count overflow"))?;
        }
        Ok(())
    }

    pub(super) fn add_to_scorer(&self, scorer: &mut MemBM25Scorer) -> Result<()> {
        scorer.num_docs = scorer
            .num_docs
            .checked_add(self.doc_count)
            .ok_or_else(|| Error::internal("residual BM25 document count overflow"))?;
        scorer.total_tokens = scorer
            .total_tokens
            .checked_add(self.total_tokens)
            .ok_or_else(|| Error::internal("residual BM25 total token count overflow"))?;
        for (token, df) in &self.token_docs {
            let current = scorer.token_docs.entry(token.clone()).or_default();
            *current = current
                .checked_add(*df)
                .ok_or_else(|| Error::internal("residual BM25 term document count overflow"))?;
        }
        Ok(())
    }
}
