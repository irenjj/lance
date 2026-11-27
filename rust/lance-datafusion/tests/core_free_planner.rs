// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use arrow_array::{BooleanArray, cast::AsArray, record_batch};
use datafusion_expr::Expr;
use lance_datafusion::planner::Planner;

#[test]
fn modular_planner_evaluates_filter_with_udf_and_like_escape() {
    let batch = record_batch!(("label", Utf8, ["A%", "Ab", "X%"])).unwrap();
    let planner = Planner::new(batch.schema());

    let filter = planner
        .parse_filter(r"lower(label) LIKE 'a\%' ESCAPE '\'")
        .unwrap();
    let optimized = planner.optimize_expr(filter).unwrap();
    let physical = planner.create_physical_expr(&optimized).unwrap();
    let actual = physical.evaluate(&batch).unwrap().into_array(3).unwrap();

    assert_eq!(
        actual.as_boolean(),
        &BooleanArray::from(vec![true, false, false])
    );
}

#[test]
fn modular_planner_preserves_custom_like_escape_in_logical_expression() {
    let batch = record_batch!(("label", Utf8, ["A%"])).unwrap();
    let planner = Planner::new(batch.schema());

    let filter = planner.parse_filter("label LIKE 'A!%' ESCAPE '!'").unwrap();
    let Expr::Like(like) = filter else {
        panic!("expected LIKE expression, got {filter:?}");
    };
    assert_eq!(like.escape_char, Some('!'));
}
