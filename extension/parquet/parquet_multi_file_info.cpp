#include "parquet_multi_file_info.hpp"
#include "duckdb/common/multi_file/multi_file_function.hpp"
#include "duckdb/parser/parsed_data/create_table_function_info.hpp"
#include "duckdb/common/serializer/serializer.hpp"
#include "duckdb/common/serializer/deserializer.hpp"
#include "parquet_crypto.hpp"
#include "duckdb/function/table_function.hpp"
#include "duckdb/storage/statistics/numeric_stats.hpp"
#include "parquet_column_schema.hpp"

namespace duckdb {

struct ParquetReadBindData : public TableFunctionData {
	// These come from the initial_reader, but need to be stored in case the initial_reader is removed by a filter
	idx_t initial_file_cardinality;
	idx_t initial_file_row_groups;
	idx_t explicit_cardinality = 0; // can be set to inject exterior cardinality knowledge (e.g. from a data lake)
	unique_ptr<ParquetFileReaderOptions> options;

	ParquetOptions &GetParquetOptions() {
		return options->options;
	}
	const ParquetOptions &GetParquetOptions() const {
		return options->options;
	}

	unique_ptr<FunctionData> Copy() const override {
		auto result = make_uniq<ParquetReadBindData>();
		result->initial_file_cardinality = initial_file_cardinality;
		result->initial_file_row_groups = initial_file_row_groups;
		result->explicit_cardinality = explicit_cardinality;
		result->options = make_uniq<ParquetFileReaderOptions>(options->options);
		return std::move(result);
	}
};

struct ParquetReadGlobalState : public GlobalTableFunctionState {
	explicit ParquetReadGlobalState(optional_ptr<const PhysicalOperator> op_p)
	    : row_group_index(0), batch_index(0), op(op_p) {
	}
	//! Index of row group within file currently up for scanning
	idx_t row_group_index;
	//! Batch index of the next row group to be scanned
	idx_t batch_index;
	//! (Optional) pointer to physical operator performing the scan
	optional_ptr<const PhysicalOperator> op;
};

struct ParquetReadLocalState : public LocalTableFunctionState {
	ParquetReaderScanState scan_state;
};

static void ParseFileRowNumberOption(MultiFileReaderBindData &bind_data, ParquetOptions &options,
                                     vector<LogicalType> &return_types, vector<string> &names) {
	if (options.file_row_number) {
		if (StringUtil::CIFind(names, "file_row_number") != DConstants::INVALID_INDEX) {
			throw BinderException(
			    "Using file_row_number option on file with column named file_row_number is not supported");
		}

		return_types.emplace_back(LogicalType::BIGINT);
		names.emplace_back("file_row_number");
	}
}

static void BindSchema(ClientContext &context, vector<LogicalType> &return_types, vector<string> &names,
                       MultiFileBindData &bind_data) {
	auto &parquet_bind = bind_data.bind_data->Cast<ParquetReadBindData>();
	auto &options = parquet_bind.GetParquetOptions();
	D_ASSERT(!options.schema.empty());

	auto &file_options = bind_data.file_options;
	if (file_options.union_by_name || file_options.hive_partitioning) {
		throw BinderException("Parquet schema cannot be combined with union_by_name=true or hive_partitioning=true");
	}
	auto &reader_bind = bind_data.reader_bind;

	vector<string> schema_col_names;
	vector<LogicalType> schema_col_types;
	schema_col_names.reserve(options.schema.size());
	schema_col_types.reserve(options.schema.size());
	bool match_by_field_id;
	if (!options.schema.empty()) {
		auto &column = options.schema[0];
		if (column.identifier.type().id() == LogicalTypeId::INTEGER) {
			match_by_field_id = true;
		} else {
			match_by_field_id = false;
		}
	} else {
		match_by_field_id = false;
	}

	for (idx_t i = 0; i < options.schema.size(); i++) {
		const auto &column = options.schema[i];
		schema_col_names.push_back(column.name);
		schema_col_types.push_back(column.type);

		auto res = MultiFileColumnDefinition(column.name, column.type);
		res.identifier = column.identifier;
#ifdef DEBUG
		if (match_by_field_id) {
			D_ASSERT(res.identifier.type().id() == LogicalTypeId::INTEGER);
		} else {
			D_ASSERT(res.identifier.type().id() == LogicalTypeId::VARCHAR);
		}
#endif

		res.default_expression = make_uniq<ConstantExpression>(column.default_value);
		reader_bind.schema.emplace_back(res);
	}
	ParseFileRowNumberOption(reader_bind, options, return_types, names);
	if (options.file_row_number) {
		MultiFileColumnDefinition res("file_row_number", LogicalType::BIGINT);
		res.identifier = Value::INTEGER(MultiFileReader::ORDINAL_FIELD_ID);
		schema_col_names.push_back(res.name);
		schema_col_types.push_back(res.type);
		reader_bind.schema.emplace_back(res);
	}

	if (match_by_field_id) {
		reader_bind.mapping = MultiFileColumnMappingMode::BY_FIELD_ID;
	} else {
		reader_bind.mapping = MultiFileColumnMappingMode::BY_NAME;
	}

	// perform the binding on the obtained set of names + types
	bind_data.multi_file_reader->BindOptions(file_options, *bind_data.file_list, schema_col_types, schema_col_names,
	                                         reader_bind);

	names = schema_col_names;
	return_types = schema_col_types;
	D_ASSERT(names.size() == return_types.size());
}

unique_ptr<MultiFileReaderInterface> ParquetMultiFileInfo::CreateInterface(ClientContext &context) {
	return make_uniq<ParquetMultiFileInfo>();
}

void ParquetMultiFileInfo::BindReader(ClientContext &context, vector<LogicalType> &return_types, vector<string> &names,
                                      MultiFileBindData &bind_data) {
	auto &parquet_bind = bind_data.bind_data->Cast<ParquetReadBindData>();
	auto &options = parquet_bind.GetParquetOptions();
	if (!options.schema.empty()) {
		BindSchema(context, return_types, names, bind_data);
	} else {
		bind_data.reader_bind =
		    bind_data.multi_file_reader->BindReader(context, return_types, names, *bind_data.file_list, bind_data,
		                                            *parquet_bind.options, bind_data.file_options);
	}
}

static bool GetBooleanArgument(const string &key, const vector<Value> &option_values) {
	if (option_values.empty()) {
		return true;
	}
	Value boolean_value;
	string error_message;
	if (!option_values[0].DefaultTryCastAs(LogicalType::BOOLEAN, boolean_value, &error_message)) {
		throw InvalidInputException("Unable to cast \"%s\" to BOOLEAN for Parquet option \"%s\"",
		                            option_values[0].ToString(), key);
	}
	return BooleanValue::Get(boolean_value);
}

static bool ParquetScanPushdownExpression(ClientContext &context, const LogicalGet &get, Expression &expr) {
	return true;
}

static void VerifyParquetSchemaParameter(const Value &schema) {
	LogicalType::MAP(LogicalType::BLOB, LogicalType::STRUCT({{{"name", LogicalType::VARCHAR},
	                                                          {"type", LogicalType::VARCHAR},
	                                                          {"default_value", LogicalType::VARCHAR}}}));
	auto &map_type = schema.type();
	if (map_type.id() != LogicalTypeId::MAP) {
		throw InvalidInputException("'schema' expects a value of type MAP, not %s",
		                            LogicalTypeIdToString(map_type.id()));
	}
	auto &key_type = MapType::KeyType(map_type);
	auto &value_type = MapType::ValueType(map_type);

	if (value_type.id() != LogicalTypeId::STRUCT) {
		throw InvalidInputException("'schema' expects a STRUCT as the value type of the map");
	}
	auto &children = StructType::GetChildTypes(value_type);
	if (children.size() < 3) {
		throw InvalidInputException(
		    "'schema' expects the STRUCT to have 3 children, 'name', 'type' and 'default_value");
	}
	if (!StringUtil::CIEquals(children[0].first, "name")) {
		throw InvalidInputException("'schema' expects the first field of the struct to be called 'name'");
	}
	if (children[0].second.id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("'schema' expects the 'name' field to be of type VARCHAR, not %s",
		                            LogicalTypeIdToString(children[0].second.id()));
	}
	if (!StringUtil::CIEquals(children[1].first, "type")) {
		throw InvalidInputException("'schema' expects the second field of the struct to be called 'type'");
	}
	if (children[1].second.id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException("'schema' expects the 'type' field to be of type VARCHAR, not %s",
		                            LogicalTypeIdToString(children[1].second.id()));
	}
	if (!StringUtil::CIEquals(children[2].first, "default_value")) {
		throw InvalidInputException("'schema' expects the third field of the struct to be called 'default_value'");
	}
	//! NOTE: default_value can be any type

	if (key_type.id() != LogicalTypeId::INTEGER && key_type.id() != LogicalTypeId::VARCHAR) {
		throw InvalidInputException(
		    "'schema' expects the value type of the map to be either INTEGER or VARCHAR, not %s",
		    LogicalTypeIdToString(key_type.id()));
	}
}

static void ParquetScanSerialize(Serializer &serializer, const optional_ptr<FunctionData> bind_data_p,
                                 const TableFunction &function) {
	auto &bind_data = bind_data_p->Cast<MultiFileBindData>();
	auto &parquet_data = bind_data.bind_data->Cast<ParquetReadBindData>();

	vector<string> files;
	for (auto &file : bind_data.file_list->GetAllFiles()) {
		files.emplace_back(file.path);
	}
	serializer.WriteProperty(100, "files", files);
	serializer.WriteProperty(101, "types", bind_data.types);
	serializer.WriteProperty(102, "names", bind_data.names);
	ParquetOptionsSerialization serialization(parquet_data.GetParquetOptions(), bind_data.file_options);
	serializer.WriteProperty(103, "parquet_options", serialization);
	if (serializer.ShouldSerialize(3)) {
		serializer.WriteProperty(104, "table_columns", bind_data.table_columns);
	}
}

static unique_ptr<FunctionData> ParquetScanDeserialize(Deserializer &deserializer, TableFunction &function) {
	auto &context = deserializer.Get<ClientContext &>();
	auto files = deserializer.ReadProperty<vector<string>>(100, "files");
	auto types = deserializer.ReadProperty<vector<LogicalType>>(101, "types");
	auto names = deserializer.ReadProperty<vector<string>>(102, "names");
	auto serialization = deserializer.ReadProperty<ParquetOptionsSerialization>(103, "parquet_options");
	auto table_columns =
	    deserializer.ReadPropertyWithExplicitDefault<vector<string>>(104, "table_columns", vector<string> {});

	vector<Value> file_path;
	for (auto &path : files) {
		file_path.emplace_back(path);
	}
	FileGlobInput input(FileGlobOptions::FALLBACK_GLOB, "parquet");

	auto multi_file_reader = MultiFileReader::Create(function);
	auto file_list = multi_file_reader->CreateFileList(context, Value::LIST(LogicalType::VARCHAR, file_path), input);
	auto parquet_options = make_uniq<ParquetFileReaderOptions>(std::move(serialization.parquet_options));
	auto interface = make_uniq<ParquetMultiFileInfo>();
	auto bind_data = MultiFileFunction<ParquetMultiFileInfo>::MultiFileBindInternal(
	    context, std::move(multi_file_reader), std::move(file_list), types, names,
	    std::move(serialization.file_options), std::move(parquet_options), std::move(interface));
	bind_data->Cast<MultiFileBindData>().table_columns = std::move(table_columns);
	return bind_data;
}

static vector<column_t> ParquetGetRowIdColumns(ClientContext &context, optional_ptr<FunctionData> bind_data) {
	vector<column_t> result;
	result.emplace_back(MultiFileReader::COLUMN_IDENTIFIER_FILE_INDEX);
	result.emplace_back(MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER);
	return result;
}

//===--------------------------------------------------------------------===//
// Helper: Validate row group ordering within a single file's metadata
//===--------------------------------------------------------------------===//
static bool ValidateSingleFileRowGroupOrdering(const duckdb_parquet::FileMetaData &metadata,
                                               const ParquetColumnSchema &column_schema,
                                               const ParquetOptions &parquet_options, const SortingColumn &sort_col) {
	for (idx_t rg_idx = 0; rg_idx + 1 < metadata.row_groups.size(); rg_idx++) {
		auto &current_rg = metadata.row_groups[rg_idx];
		auto &next_rg = metadata.row_groups[rg_idx + 1];

		auto current_stats = column_schema.Stats(metadata, parquet_options, rg_idx, current_rg.columns);
		auto next_stats = column_schema.Stats(metadata, parquet_options, rg_idx + 1, next_rg.columns);

		if (!current_stats || !next_stats) {
			return false;
		}

		if (!NumericStats::HasMax(*current_stats) || !NumericStats::HasMin(*next_stats)) {
			return false;
		}

		Value current_max = NumericStats::Max(*current_stats);
		Value next_min = NumericStats::Min(*next_stats);

		if (sort_col.descending) {
			if (!NumericStats::HasMin(*current_stats) || !NumericStats::HasMax(*next_stats)) {
				return false;
			}
			Value current_min = NumericStats::Min(*current_stats);
			Value next_max = NumericStats::Max(*next_stats);
			if (current_min < next_max) {
				return false;
			}
		} else {
			if (current_max > next_min) {
				return false;
			}
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Helper: File statistics for sorting
//===--------------------------------------------------------------------===//
struct FileStatistics {
	idx_t file_idx;
	Value min_val;
	Value max_val;
};

//===--------------------------------------------------------------------===//
// Helper: Extract min/max statistics from each file's first row group
// Returns empty vector on failure
//===--------------------------------------------------------------------===//
static vector<FileStatistics> ExtractFileStatistics(const vector<shared_ptr<ParquetFileMetadataCache>> &file_caches,
                                                    const ParquetColumnSchema &column_schema,
                                                    const ParquetOptions &parquet_options) {
	vector<FileStatistics> result;
	for (idx_t i = 0; i < file_caches.size(); i++) {
		auto &metadata = *file_caches[i]->metadata;
		if (metadata.row_groups.empty()) {
			return {};
		}

		auto &first_rg = metadata.row_groups[0];
		auto stats = column_schema.Stats(metadata, parquet_options, 0, first_rg.columns);
		if (!stats || !NumericStats::HasMin(*stats) || !NumericStats::HasMax(*stats)) {
			return {};
		}

		result.push_back({i, NumericStats::Min(*stats), NumericStats::Max(*stats)});
	}
	return result;
}

//===--------------------------------------------------------------------===//
// Helper: Sort files by their statistics to determine optimal ordering
//===--------------------------------------------------------------------===//
static vector<idx_t> SortFilesByStatistics(vector<FileStatistics> &stats, bool descending) {
	if (descending) {
		// DESC: sort by max value descending (highest max first)
		std::sort(stats.begin(), stats.end(),
		          [](const FileStatistics &a, const FileStatistics &b) { return a.max_val > b.max_val; });
	} else {
		// ASC: sort by min value ascending (lowest min first)
		std::sort(stats.begin(), stats.end(),
		          [](const FileStatistics &a, const FileStatistics &b) { return a.min_val < b.min_val; });
	}

	vector<idx_t> order;
	for (auto &s : stats) {
		order.push_back(s.file_idx);
	}
	return order;
}

//===--------------------------------------------------------------------===//
// Helper: Validate ordering between files (last RG of file[i] vs first RG of file[i+1])
//===--------------------------------------------------------------------===//
static bool ValidateCrossFileOrdering(const vector<shared_ptr<ParquetFileMetadataCache>> &file_caches,
                                      const vector<idx_t> &sorted_order, const ParquetColumnSchema &column_schema,
                                      const ParquetOptions &parquet_options, const SortingColumn &sort_col) {
	for (idx_t i = 0; i + 1 < sorted_order.size(); i++) {
		idx_t current_idx = sorted_order[i];
		idx_t next_idx = sorted_order[i + 1];
		auto &current_meta = *file_caches[current_idx]->metadata;
		auto &next_meta = *file_caches[next_idx]->metadata;

		if (current_meta.row_groups.empty() || next_meta.row_groups.empty()) {
			return false;
		}

		// Get last row group of current file
		idx_t last_rg_idx = current_meta.row_groups.size() - 1;
		auto &last_rg = current_meta.row_groups[last_rg_idx];

		// Get first row group of next file
		auto &first_rg = next_meta.row_groups[0];

		// Get statistics
		auto last_stats = column_schema.Stats(current_meta, parquet_options, last_rg_idx, last_rg.columns);
		auto first_stats = column_schema.Stats(next_meta, parquet_options, 0, first_rg.columns);

		if (!last_stats || !first_stats) {
			return false;
		}

		if (!NumericStats::HasMax(*last_stats) || !NumericStats::HasMin(*first_stats)) {
			return false;
		}

		Value last_max = NumericStats::Max(*last_stats);
		Value first_min = NumericStats::Min(*first_stats);

		if (sort_col.descending) {
			if (!NumericStats::HasMin(*last_stats) || !NumericStats::HasMax(*first_stats)) {
				return false;
			}
			Value last_min = NumericStats::Min(*last_stats);
			Value first_max = NumericStats::Max(*first_stats);
			if (last_min < first_max) {
				return false;
			}
		} else {
			if (last_max > first_min) {
				return false;
			}
		}
	}
	return true;
}

//===--------------------------------------------------------------------===//
// Helper: Compute common sorting prefix across all row groups in metadata
//===--------------------------------------------------------------------===//
static void ComputeCommonSortingPrefix(const duckdb_parquet::FileMetaData &metadata, vector<SortingColumn> &result,
                                       bool &initialized) {
	for (idx_t rg_idx = 0; rg_idx < metadata.row_groups.size(); rg_idx++) {
		auto &rg = metadata.row_groups[rg_idx];
		if (!rg.__isset.sorting_columns || rg.sorting_columns.empty()) {
			// No sorting info for this row group - common prefix is empty
			result.clear();
			initialized = true;
			return;
		}

		if (!initialized) {
			// Initialize from first row group encountered
			for (auto &sort_col : rg.sorting_columns) {
				result.emplace_back(static_cast<column_t>(sort_col.column_idx), sort_col.descending,
				                    sort_col.nulls_first);
			}
			initialized = true;
		} else {
			// Find the common prefix length
			idx_t common_len = MinValue(result.size(), rg.sorting_columns.size());
			idx_t prefix_len = 0;
			for (idx_t j = 0; j < common_len; j++) {
				auto &existing = result[j];
				auto &current = rg.sorting_columns[j];
				if (existing.column_idx == static_cast<column_t>(current.column_idx) &&
				    existing.descending == current.descending && existing.nulls_first == current.nulls_first) {
					prefix_len++;
				} else {
					break;
				}
			}
			// Truncate to the common prefix
			result.resize(prefix_len);
		}

		// If common prefix is now empty, no point continuing
		if (result.empty() && initialized) {
			return;
		}
	}
}

//===--------------------------------------------------------------------===//
// ParquetGetSortingOrder - Main function
//===--------------------------------------------------------------------===//
static vector<SortingColumn> ParquetGetSortingOrder(ClientContext &context,
                                                    optional_ptr<const FunctionData> bind_data_p) {
	auto &bind_data = bind_data_p->Cast<MultiFileBindData>();
	vector<SortingColumn> result;

	auto expand_result = bind_data.file_list->GetExpandResult();
	if (expand_result == FileExpandResult::NO_FILES) {
		return result;
	}

	if (!bind_data.initial_reader) {
		return result;
	}

	auto &initial_reader = bind_data.initial_reader->Cast<ParquetReader>();
	auto &parquet_options = initial_reader.parquet_options;

	// Collect metadata for all files
	vector<shared_ptr<ParquetFileMetadataCache>> file_caches;

	if (bind_data.file_list->GetExpandResult() == FileExpandResult::SINGLE_FILE) {
		// Single file - use metadata from initial_reader directly
		file_caches.push_back(initial_reader.metadata);
	} else {
		// Multiple files - require metadata caching
		if (!ParquetReader::MetadataCacheEnabled(context)) {
			return result;
		}
		// Collect cached metadata for all files
		// Note: We skip cache validation for sorting order because:
		// 1. Sorting metadata is read-only optimization information
		// 2. If the cache is stale, we might miss an optimization but correctness is preserved
		// 3. Parquet sorting order is immutable - a file change would invalidate the entire cache
		for (auto &file : bind_data.file_list->Files()) {
			auto cache_entry = ParquetReader::GetMetadataCacheEntry(context, file);
			if (!cache_entry) {
				// No cache entry - bail
				return result;
			}
			file_caches.push_back(std::move(cache_entry));
		}
	}

	if (file_caches.empty()) {
		return result;
	}

	// Compute common sorting prefix across ALL row groups in ALL files
	bool initialized = false;
	for (idx_t file_idx = 0; file_idx < file_caches.size(); file_idx++) {
		auto &metadata = *file_caches[file_idx]->metadata;
		ComputeCommonSortingPrefix(metadata, result, initialized);
		if (result.empty() && initialized) {
			// Common prefix became empty - no point continuing
			return result;
		}
	}

	if (result.empty()) {
		return result;
	}

	// Validate ordering using statistics on the first sorted column
	// For composite sorts like (a, b), only the first column needs global ordering
	if (!initial_reader.root_schema) {
		result.clear();
		return result;
	}

	auto &sort_col = result[0];
	auto table_col_idx = sort_col.column_idx;

	if (table_col_idx >= initial_reader.root_schema->children.size()) {
		result.clear();
		return result;
	}
	auto &column_schema = initial_reader.root_schema->children[table_col_idx];

	// Validate row group ordering within each file
	for (auto &cache : file_caches) {
		auto &metadata = *cache->metadata;
		if (metadata.row_groups.size() > 1) {
			if (!ValidateSingleFileRowGroupOrdering(metadata, column_schema, parquet_options, sort_col)) {
				result.clear();
				return result;
			}
		}
	}

	// Validate ordering between files (if multiple files)
	if (file_caches.size() > 1) {
		// Extract statistics from each file to determine optimal ordering
		auto file_stats = ExtractFileStatistics(file_caches, column_schema, parquet_options);
		if (file_stats.empty()) {
			result.clear();
			return result;
		}

		// Sort files by their statistics to find the correct order
		// Note: FinalizeBindData attempts to reorder files, but may fail if cache isn't populated
		// So we compute the sorted order here to be robust
		auto sorted_order = SortFilesByStatistics(file_stats, sort_col.descending);

		// Validate cross-file ordering using the sorted order
		if (!ValidateCrossFileOrdering(file_caches, sorted_order, column_schema, parquet_options, sort_col)) {
			result.clear();
			return result;
		}
	}

	return result;
}

static vector<PartitionStatistics> ParquetGetPartitionStats(ClientContext &context, GetPartitionStatsInput &input) {
	auto &bind_data = input.bind_data->Cast<MultiFileBindData>();
	vector<PartitionStatistics> result;
	if (bind_data.file_list->GetExpandResult() == FileExpandResult::SINGLE_FILE && bind_data.initial_reader) {
		// we have read the metadata - get the partitions for this reader
		auto &reader = bind_data.initial_reader->Cast<ParquetReader>();
		reader.GetPartitionStats(result);
		return result;
	}
	// if we are reading multiple files - we check if we have caching enabled
	if (!ParquetReader::MetadataCacheEnabled(context)) {
		// no caching - bail
		return result;
	}
	// caching is enabled - check if we have ALL of the metadata cached
	vector<shared_ptr<ParquetFileMetadataCache>> caches;
	for (auto &file : bind_data.file_list->Files()) {
		auto metadata_entry = ParquetReader::GetMetadataCacheEntry(context, file);
		if (!metadata_entry) {
			// no cache entry found
			return result;
		}
		// check if the file has any deletes
		if (file.extended_info) {
			auto entry = file.extended_info->options.find("has_deletes");
			if (entry != file.extended_info->options.end()) {
				if (BooleanValue::Get(entry->second)) {
					// the file has deletes - skip emitting partition stats
					// FIXME: we could emit partition stats but set count to `COUNT_APPROXIMATE` instead of
					// `COUNT_EXACT`
					return result;
				}
			}
		}

		// check if the cache is valid based ONLY on the OpenFileInfo (do not do any file system requests here)
		const auto is_valid = metadata_entry->IsValid(file, context);
		if (is_valid != ParquetCacheValidity::VALID) {
			return result;
		}
		caches.push_back(std::move(metadata_entry));
	}
	// all caches are valid! we can return the partition stats
	for (auto &cache : caches) {
		ParquetReader::GetPartitionStats(*cache->metadata, result);
	}
	return result;
}

TableFunctionSet ParquetScanFunction::GetFunctionSet() {
	MultiFileFunction<ParquetMultiFileInfo> table_function("parquet_scan");
	table_function.named_parameters["binary_as_string"] = LogicalType::BOOLEAN;
	table_function.named_parameters["file_row_number"] = LogicalType::BOOLEAN;
	table_function.named_parameters["debug_use_openssl"] = LogicalType::BOOLEAN;
	table_function.named_parameters["compression"] = LogicalType::VARCHAR;
	table_function.named_parameters["explicit_cardinality"] = LogicalType::UBIGINT;
	table_function.named_parameters["schema"] = LogicalTypeId::ANY;
	table_function.named_parameters["encryption_config"] = LogicalTypeId::ANY;
	table_function.named_parameters["parquet_version"] = LogicalType::VARCHAR;
	table_function.named_parameters["can_have_nan"] = LogicalType::BOOLEAN;
	table_function.statistics = MultiFileFunction<ParquetMultiFileInfo>::MultiFileScanStats;
	table_function.serialize = ParquetScanSerialize;
	table_function.deserialize = ParquetScanDeserialize;
	table_function.get_row_id_columns = ParquetGetRowIdColumns;
	table_function.pushdown_expression = ParquetScanPushdownExpression;
	table_function.get_partition_stats = ParquetGetPartitionStats;
	table_function.get_sorting_order = ParquetGetSortingOrder;
	table_function.filter_pushdown = true;
	table_function.filter_prune = true;
	table_function.late_materialization = true;

	return MultiFileReader::CreateFunctionSet(static_cast<TableFunction>(table_function));
}

unique_ptr<BaseFileReaderOptions> ParquetMultiFileInfo::InitializeOptions(ClientContext &context,
                                                                          optional_ptr<TableFunctionInfo> info) {
	return make_uniq<ParquetFileReaderOptions>(context);
}

bool ParquetMultiFileInfo::ParseCopyOption(ClientContext &context, const string &key, const vector<Value> &values,
                                           BaseFileReaderOptions &file_options, vector<string> &expected_names,
                                           vector<LogicalType> &expected_types) {
	auto &parquet_options = file_options.Cast<ParquetFileReaderOptions>();
	auto &options = parquet_options.options;
	if (key == "compression" || key == "codec" || key == "row_group_size") {
		// CODEC/COMPRESSION and ROW_GROUP_SIZE options have no effect on parquet read.
		// These options are determined from the file.
		return true;
	}
	if (key == "binary_as_string") {
		options.binary_as_string = GetBooleanArgument(key, values);
		return true;
	}
	if (key == "file_row_number") {
		options.file_row_number = GetBooleanArgument(key, values);
		return true;
	}
	if (key == "debug_use_openssl") {
		options.debug_use_openssl = GetBooleanArgument(key, values);
		return true;
	}
	if (key == "encryption_config") {
		if (values.size() != 1) {
			throw BinderException("Parquet encryption_config cannot be empty!");
		}
		options.encryption_config = ParquetEncryptionConfig::Create(context, values[0]);
		return true;
	}
	if (key == "can_have_nan") {
		if (values.size() != 1) {
			throw BinderException("Parquet can_have_nan cannot be empty!");
		}
		options.can_have_nan = GetBooleanArgument(key, values);
		return true;
	}
	return false;
}

bool ParquetMultiFileInfo::ParseOption(ClientContext &context, const string &original_key, const Value &val,
                                       MultiFileOptions &file_options, BaseFileReaderOptions &base_options) {
	auto &parquet_options = base_options.Cast<ParquetFileReaderOptions>();
	auto &options = parquet_options.options;
	auto key = StringUtil::Lower(original_key);
	if (val.IsNull()) {
		throw BinderException("Cannot use NULL as argument to %s", original_key);
	}
	if (key == "compression") {
		// COMPRESSION has no effect on parquet read.
		// These options are determined from the file.
		return true;
	}
	if (key == "binary_as_string") {
		options.binary_as_string = BooleanValue::Get(val);
		return true;
	}
	if (key == "file_row_number") {
		options.file_row_number = BooleanValue::Get(val);
		return true;
	}
	if (key == "debug_use_openssl") {
		options.debug_use_openssl = BooleanValue::Get(val);
		return true;
	}
	if (key == "can_have_nan") {
		options.can_have_nan = BooleanValue::Get(val);
		return true;
	}
	if (key == "schema") {
		// Argument is a map that defines the schema
		const auto &schema_value = val;
		VerifyParquetSchemaParameter(schema_value);
		const auto column_values = ListValue::GetChildren(schema_value);
		if (column_values.empty()) {
			throw BinderException("Parquet schema cannot be empty");
		}
		options.schema.reserve(column_values.size());
		for (idx_t i = 0; i < column_values.size(); i++) {
			options.schema.emplace_back(ParquetColumnDefinition::FromSchemaValue(context, column_values[i]));
		}
		file_options.auto_detect_hive_partitioning = false;
		return true;
	}
	if (key == "explicit_cardinality") {
		options.explicit_cardinality = UBigIntValue::Get(val);
		return true;
	}
	if (key == "encryption_config") {
		options.encryption_config = ParquetEncryptionConfig::Create(context, val);
		return true;
	}
	return false;
}

unique_ptr<TableFunctionData> ParquetMultiFileInfo::InitializeBindData(MultiFileBindData &multi_file_data,
                                                                       unique_ptr<BaseFileReaderOptions> options_p) {
	auto result = make_uniq<ParquetReadBindData>();
	// Set the explicit cardinality if requested
	result->options = unique_ptr_cast<BaseFileReaderOptions, ParquetFileReaderOptions>(std::move(options_p));
	auto &parquet_options = result->GetParquetOptions();
	if (parquet_options.explicit_cardinality) {
		auto file_count = multi_file_data.file_list->GetTotalFileCount();
		result->explicit_cardinality = parquet_options.explicit_cardinality;
		result->initial_file_cardinality = result->explicit_cardinality / (file_count ? file_count : 1);
	}
	return std::move(result);
}

void ParquetMultiFileInfo::GetBindInfo(const TableFunctionData &bind_data_p, BindInfo &info) {
	auto &bind_data = bind_data_p.Cast<ParquetReadBindData>();
	auto &parquet_options = bind_data.GetParquetOptions();
	info.type = ScanType::PARQUET;
	info.InsertOption("binary_as_string", Value::BOOLEAN(parquet_options.binary_as_string));
	info.InsertOption("file_row_number", Value::BOOLEAN(parquet_options.file_row_number));
	info.InsertOption("debug_use_openssl", Value::BOOLEAN(parquet_options.debug_use_openssl));
}

optional_idx ParquetMultiFileInfo::MaxThreads(const MultiFileBindData &bind_data_p,
                                              const MultiFileGlobalState &global_state,
                                              FileExpandResult expand_result) {
	if (expand_result == FileExpandResult::MULTIPLE_FILES) {
		// always launch max threads if we are reading multiple files
		return optional_idx();
	}
	auto &bind_data = bind_data_p.bind_data->Cast<ParquetReadBindData>();
	return MaxValue(bind_data.initial_file_row_groups, static_cast<idx_t>(1));
}

//! Reorders files in the file list so that they are sorted by their sorting column statistics.
//! This ensures that when ORDER BY is eliminated, reading files in list order produces sorted output.
static void ReorderFilesBySortingColumn(ClientContext &context, MultiFileBindData &bind_data) {
	// Only applicable for multiple files with caching enabled
	if (bind_data.file_list->GetExpandResult() != FileExpandResult::MULTIPLE_FILES) {
		return;
	}
	if (!ParquetReader::MetadataCacheEnabled(context)) {
		return;
	}
	if (!bind_data.initial_reader) {
		return;
	}

	auto &initial_reader = bind_data.initial_reader->Cast<ParquetReader>();
	auto &parquet_options = initial_reader.parquet_options;

	// Collect cached metadata for all files
	vector<shared_ptr<ParquetFileMetadataCache>> file_caches;
	vector<OpenFileInfo> files;
	for (auto &file : bind_data.file_list->Files()) {
		auto cache_entry = ParquetReader::GetMetadataCacheEntry(context, file);
		if (!cache_entry) {
			return; // No cache, can't reorder
		}
		file_caches.push_back(std::move(cache_entry));
		files.push_back(file);
	}

	if (file_caches.empty()) {
		return;
	}

	// Compute common sorting prefix across all files
	vector<SortingColumn> sorting_cols;
	bool initialized = false;
	for (auto &cache : file_caches) {
		ComputeCommonSortingPrefix(*cache->metadata, sorting_cols, initialized);
		if (sorting_cols.empty() && initialized) {
			return; // No common sorting
		}
	}

	if (sorting_cols.empty()) {
		return;
	}

	// Get column schema for statistics extraction
	if (!initial_reader.root_schema) {
		return;
	}
	auto &sort_col = sorting_cols[0];
	if (sort_col.column_idx >= initial_reader.root_schema->children.size()) {
		return;
	}
	auto &column_schema = initial_reader.root_schema->children[sort_col.column_idx];

	// Extract statistics and compute sorted order
	auto file_stats = ExtractFileStatistics(file_caches, column_schema, parquet_options);
	if (file_stats.empty()) {
		return;
	}
	auto sorted_order = SortFilesByStatistics(file_stats, sort_col.descending);

	// Check if already in order - no need to reorder
	bool already_sorted = true;
	for (idx_t i = 0; i < sorted_order.size(); i++) {
		if (sorted_order[i] != i) {
			already_sorted = false;
			break;
		}
	}
	if (already_sorted) {
		return;
	}

	// Reorder the files according to sorted_order
	vector<OpenFileInfo> reordered_files;
	reordered_files.reserve(sorted_order.size());
	for (idx_t sorted_idx : sorted_order) {
		reordered_files.push_back(std::move(files[sorted_idx]));
	}

	// Replace the file list with reordered version
	bind_data.file_list = make_shared_ptr<SimpleMultiFileList>(std::move(reordered_files));
}

void ParquetMultiFileInfo::FinalizeBindData(ClientContext &context, MultiFileBindData &multi_file_data) {
	auto &bind_data = multi_file_data.bind_data->Cast<ParquetReadBindData>();
	if (multi_file_data.initial_reader) {
		auto &initial_reader = multi_file_data.initial_reader->Cast<ParquetReader>();
		bind_data.initial_file_cardinality = initial_reader.NumRows();
		bind_data.initial_file_row_groups = initial_reader.NumRowGroups();
		bind_data.options->options = initial_reader.parquet_options;
	}

	// Reorder files for sort elimination optimization
	ReorderFilesBySortingColumn(context, multi_file_data);
}

unique_ptr<NodeStatistics> ParquetMultiFileInfo::GetCardinality(const MultiFileBindData &bind_data_p,
                                                                idx_t file_count) {
	auto &bind_data = bind_data_p.bind_data->Cast<ParquetReadBindData>();
	if (bind_data.explicit_cardinality) {
		return make_uniq<NodeStatistics>(bind_data.explicit_cardinality);
	}
	return make_uniq<NodeStatistics>(MaxValue(bind_data.initial_file_cardinality, (idx_t)1) * file_count);
}

unique_ptr<BaseStatistics> ParquetReader::GetStatistics(ClientContext &context, const string &name) {
	return ReadStatistics(name);
}

double ParquetReader::GetProgressInFile(ClientContext &context) {
	auto read_rows = rows_read.load();
	return 100.0 * (static_cast<double>(read_rows) / static_cast<double>(NumRows()));
}

void ParquetMultiFileInfo::GetVirtualColumns(ClientContext &, MultiFileBindData &, virtual_column_map_t &result) {
	result.insert(make_pair(MultiFileReader::COLUMN_IDENTIFIER_FILE_ROW_NUMBER,
	                        TableColumn("file_row_number", LogicalType::BIGINT)));
}

shared_ptr<BaseFileReader> ParquetMultiFileInfo::CreateReader(ClientContext &context, GlobalTableFunctionState &,
                                                              BaseUnionData &union_data_p,
                                                              const MultiFileBindData &bind_data_p) {
	auto &union_data = union_data_p.Cast<ParquetUnionData>();
	return make_shared_ptr<ParquetReader>(context, union_data.file, union_data.options, union_data.metadata);
}

shared_ptr<BaseFileReader> ParquetMultiFileInfo::CreateReader(ClientContext &context, GlobalTableFunctionState &,
                                                              const OpenFileInfo &file, idx_t file_idx,
                                                              const MultiFileBindData &multi_bind_data) {
	auto &bind_data = multi_bind_data.bind_data->Cast<ParquetReadBindData>();
	return make_shared_ptr<ParquetReader>(context, file, bind_data.GetParquetOptions());
}

shared_ptr<BaseFileReader> ParquetMultiFileInfo::CreateReader(ClientContext &context, const OpenFileInfo &file,
                                                              BaseFileReaderOptions &options_p,
                                                              const MultiFileOptions &) {
	auto &options = options_p.Cast<ParquetFileReaderOptions>();
	return make_shared_ptr<ParquetReader>(context, file, options.options);
}

shared_ptr<BaseUnionData> ParquetReader::GetUnionData(idx_t file_idx) {
	auto result = make_uniq<ParquetUnionData>(file);
	for (auto &column : columns) {
		result->names.push_back(column.name);
		result->types.push_back(column.type);
	}
	if (file_idx == 0) {
		result->options = parquet_options;
		result->metadata = metadata;
		result->reader = shared_from_this();
	} else {
		result->options = std::move(parquet_options);
		result->metadata = std::move(metadata);
		result->root_schema = std::move(root_schema);
	}
	return std::move(result);
}

unique_ptr<GlobalTableFunctionState> ParquetMultiFileInfo::InitializeGlobalState(ClientContext &, MultiFileBindData &,
                                                                                 MultiFileGlobalState &global_state) {
	return make_uniq<ParquetReadGlobalState>(global_state.op);
}

unique_ptr<LocalTableFunctionState> ParquetMultiFileInfo::InitializeLocalState(ExecutionContext &,
                                                                               GlobalTableFunctionState &) {
	return make_uniq<ParquetReadLocalState>();
}

bool ParquetReader::TryInitializeScan(ClientContext &context, GlobalTableFunctionState &gstate_p,
                                      LocalTableFunctionState &lstate_p) {
	auto &gstate = gstate_p.Cast<ParquetReadGlobalState>();
	auto &lstate = lstate_p.Cast<ParquetReadLocalState>();
	if (gstate.row_group_index >= NumRowGroups()) {
		// scanned all row groups in this file
		return false;
	}
	// The current reader has rowgroups left to be scanned
	vector<idx_t> group_indexes {gstate.row_group_index};
	InitializeScan(context, lstate.scan_state, group_indexes);
	gstate.row_group_index++;
	return true;
}

void ParquetReader::FinishFile(ClientContext &context, GlobalTableFunctionState &gstate_p) {
	auto &gstate = gstate_p.Cast<ParquetReadGlobalState>();
	gstate.row_group_index = 0;
}

AsyncResult ParquetReader::Scan(ClientContext &context, GlobalTableFunctionState &gstate_p,
                                LocalTableFunctionState &local_state_p, DataChunk &chunk) {
#ifdef DUCKDB_DEBUG_ASYNC_SINK_SOURCE
	{
		vector<unique_ptr<AsyncTask>> tasks = AsyncResult::GenerateTestTasks();
		if (!tasks.empty()) {
			return AsyncResult(std::move(tasks));
		}
	}
#endif

	auto &gstate = gstate_p.Cast<ParquetReadGlobalState>();
	auto &local_state = local_state_p.Cast<ParquetReadLocalState>();
	local_state.scan_state.op = gstate.op;
	return Scan(context, local_state.scan_state, chunk);
}

unique_ptr<MultiFileReaderInterface> ParquetMultiFileInfo::Copy() {
	return make_uniq<ParquetMultiFileInfo>();
}

FileGlobInput ParquetMultiFileInfo::GetGlobInput() {
	return FileGlobInput(FileGlobOptions::FALLBACK_GLOB, "parquet");
}

} // namespace duckdb
