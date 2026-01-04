#include "duckdb/optimizer/sort_elimination.hpp"

#include "duckdb/planner/operator/logical_get.hpp"
#include "duckdb/planner/operator/logical_order.hpp"
#include "duckdb/planner/operator/logical_projection.hpp"
#include "duckdb/planner/expression/bound_columnref_expression.hpp"
#include "duckdb/function/table_function.hpp"

namespace duckdb {

SortElimination::SortElimination(ClientContext &context_p) : context(context_p) {
}

unique_ptr<LogicalOperator> SortElimination::Optimize(unique_ptr<LogicalOperator> op) {
	// Recursively optimize children first
	for (auto &child : op->children) {
		child = Optimize(std::move(child));
	}

	// Check if this is an ORDER BY that can be eliminated
	if (op->type == LogicalOperatorType::LOGICAL_ORDER_BY) {
		auto &order = op->Cast<LogicalOrder>();
		if (!order.orders.empty() && !op->children.empty()) {
			if (TryEliminateSort(order, *op->children[0])) {
				// Sort can be eliminated - return the child directly
				return std::move(op->children[0]);
			}
		}
	}

	return op;
}

bool SortElimination::TryEliminateSort(LogicalOrder &order, LogicalOperator &child) {
	// Build the chain of operators from ORDER BY to GET
	vector<reference<LogicalOperator>> op_chain;
	if (!BuildOperatorChain(child, op_chain)) {
		return false;
	}

	// The last operator should be a GET
	auto &get = op_chain.back().get().Cast<LogicalGet>();

	// Check if the table function provides sorting order information
	if (!get.function.get_sorting_order) {
		return false;
	}

	// Get the sorting order from the table function
	auto sorting_order = get.function.get_sorting_order(context, get.bind_data.get());

	if (sorting_order.empty()) {
		return false;
	}

	// Check if the ORDER BY matches the sorting order
	return MatchesSortingOrder(order.orders, sorting_order, op_chain);
}

bool SortElimination::BuildOperatorChain(LogicalOperator &op, vector<reference<LogicalOperator>> &chain) {
	chain.push_back(op);

	// Allow projection between ORDER BY and GET
	if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
		if (op.children.size() != 1) {
			return false;
		}
		return BuildOperatorChain(*op.children[0], chain);
	}

	if (op.type == LogicalOperatorType::LOGICAL_GET) {
		return true;
	}

	// Other operators may change order or filter rows
	return false;
}

bool SortElimination::MatchesSortingOrder(const vector<BoundOrderByNode> &orders,
                                          const vector<SortingColumn> &sorting_order,
                                          const vector<reference<LogicalOperator>> &op_chain) {
	// ORDER BY must be a prefix of the sorting order (or exact match)
	if (orders.size() > sorting_order.size()) {
		return false;
	}

	// Get the LogicalGet from the end of the chain
	auto &get = op_chain.back().get().Cast<LogicalGet>();
	auto &column_ids = get.GetColumnIds();

	for (idx_t i = 0; i < orders.size(); i++) {
		auto &order = orders[i];
		auto &sort_col = sorting_order[i];

		// Only support simple column references
		if (order.expression->type != ExpressionType::BOUND_COLUMN_REF) {
			return false;
		}

		auto &colref = order.expression->Cast<BoundColumnRefExpression>();

		// Trace the column binding through the operator chain to find the GET column
		ColumnBinding binding = colref.binding;
		bool found_get_binding = false;

		// Traverse the chain from the first operator (closest to ORDER BY) to the GET
		for (auto &op_ref : op_chain) {
			auto &op = op_ref.get();

			if (op.type == LogicalOperatorType::LOGICAL_PROJECTION) {
				auto &proj = op.Cast<LogicalProjection>();
				// Check if the binding refers to this projection's output
				if (binding.table_index == proj.table_index) {
					// Find the expression that produces this column
					if (binding.column_index >= proj.expressions.size()) {
						return false;
					}
					auto &expr = proj.expressions[binding.column_index];
					// Must be a simple column reference to trace further
					if (expr->type != ExpressionType::BOUND_COLUMN_REF) {
						return false;
					}
					// Update binding to the input column
					binding = expr->Cast<BoundColumnRefExpression>().binding;
				}
			} else if (op.type == LogicalOperatorType::LOGICAL_GET) {
				// Check if binding refers to this GET
				if (binding.table_index == get.table_index) {
					found_get_binding = true;
				}
			}
		}

		if (!found_get_binding || binding.table_index != get.table_index) {
			return false;
		}

		// Map the column binding to the physical table column index
		if (binding.column_index >= column_ids.size()) {
			return false;
		}

		column_t table_column_idx = column_ids[binding.column_index].GetPrimaryIndex();

		if (table_column_idx != sort_col.column_idx) {
			return false;
		}

		bool order_descending = (order.type == OrderType::DESCENDING);
		if (order_descending != sort_col.descending) {
			return false;
		}

		bool order_nulls_first = (order.null_order == OrderByNullType::NULLS_FIRST);
		if (order_nulls_first != sort_col.nulls_first) {
			return false;
		}
	}

	return true;
}

} // namespace duckdb
