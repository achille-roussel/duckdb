//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/optimizer/sort_elimination.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/main/client_context.hpp"
#include "duckdb/common/constants.hpp"
#include "duckdb/function/sorting_order.hpp"

namespace duckdb {

class LogicalOperator;
class LogicalOrder;
class LogicalGet;
struct BoundOrderByNode;

//! The SortElimination optimizer eliminates redundant ORDER BY operations
//! when the input data is already sorted (e.g., from Parquet files with sorting_columns metadata)
class SortElimination {
public:
	explicit SortElimination(ClientContext &context);

	//! Optimize the plan by eliminating redundant ORDER BY operations
	unique_ptr<LogicalOperator> Optimize(unique_ptr<LogicalOperator> op);

private:
	//! Try to eliminate the sort operation if the input is already sorted
	bool TryEliminateSort(LogicalOrder &order, LogicalOperator &child);
	//! Build the chain of operators from the child to the GET
	bool BuildOperatorChain(LogicalOperator &op, vector<reference<LogicalOperator>> &chain);
	//! Check if the ORDER BY columns match the sorting order from the source
	bool MatchesSortingOrder(const vector<BoundOrderByNode> &orders, const vector<SortingColumn> &sorting_order,
	                         const vector<reference<LogicalOperator>> &op_chain);

private:
	ClientContext &context;
};

} // namespace duckdb
