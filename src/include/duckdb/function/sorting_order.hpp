//===----------------------------------------------------------------------===//
//                         DuckDB
//
// duckdb/function/sorting_order.hpp
//
//
//===----------------------------------------------------------------------===//

#pragma once

#include "duckdb/common/common.hpp"
#include "duckdb/common/vector.hpp"

namespace duckdb {

//! Represents a single column in a sorting order specification
struct SortingColumn {
	//! The column index in the table
	column_t column_idx;
	//! True if the column is sorted in descending order
	bool descending;
	//! True if nulls appear before non-null values
	bool nulls_first;

	SortingColumn() : column_idx(COLUMN_IDENTIFIER_ROW_ID), descending(false), nulls_first(false) {
	}

	SortingColumn(column_t column_idx_p, bool descending_p, bool nulls_first_p)
	    : column_idx(column_idx_p), descending(descending_p), nulls_first(nulls_first_p) {
	}

	bool operator==(const SortingColumn &other) const {
		return column_idx == other.column_idx && descending == other.descending && nulls_first == other.nulls_first;
	}
};

} // namespace duckdb
