"""Catálogo de operadores/estágios MongoDB e APIs reais da IDE (Console Jint) e do mongosh."""
from __future__ import annotations

QUERY_OPERATORS = {
    "$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$exists", "$type", "$regex", "$options",
    "$not", "$elemMatch", "$size", "$all", "$mod", "$bitsAllSet", "$bitsAnySet", "$bitsAllClear", "$bitsAnyClear",
    "$geoWithin", "$geoIntersects", "$near", "$nearSphere", "$jsonSchema",
}
LOGICAL_OPERATORS = {"$and", "$or", "$nor"}
TOP_LEVEL_QUERY = LOGICAL_OPERATORS | {"$expr", "$text", "$comment", "$where", "$jsonSchema"}

UPDATE_OPERATORS = {
    "$set", "$unset", "$inc", "$mul", "$rename", "$min", "$max", "$currentDate", "$setOnInsert",
    "$push", "$addToSet", "$pull", "$pullAll", "$pop", "$bit",
}
UPDATE_MODIFIERS = {"$each", "$position", "$slice", "$sort"}

STAGES = {
    "$match", "$group", "$project", "$sort", "$limit", "$skip", "$lookup", "$unwind", "$facet", "$set",
    "$addFields", "$unset", "$count", "$replaceRoot", "$replaceWith", "$sortByCount", "$bucket", "$bucketAuto",
    "$sample", "$search", "$searchMeta", "$vectorSearch", "$unionWith", "$graphLookup", "$densify", "$fill",
    "$setWindowFields", "$redact", "$geoNear", "$indexStats", "$collStats", "$documents", "$out", "$merge",
}
# O Console da IDE rejeita escrita a partir do cursor de leitura.
CONSOLE_FORBIDDEN_STAGES = {"$out", "$merge"}
FIRST_STAGE_ONLY = {"$search", "$searchMeta", "$vectorSearch", "$geoNear", "$indexStats", "$collStats", "$documents"}

ACCUMULATORS = {
    "$sum", "$avg", "$min", "$max", "$push", "$addToSet", "$first", "$last", "$count", "$stdDevPop",
    "$stdDevSamp", "$mergeObjects", "$top", "$bottom", "$topN", "$bottomN", "$firstN", "$lastN", "$median",
    "$percentile",
}

EXPRESSION_OPERATORS = ACCUMULATORS | {
    "$abs", "$add", "$ceil", "$divide", "$exp", "$floor", "$ln", "$log", "$log10", "$mod", "$multiply", "$pow",
    "$round", "$sqrt", "$subtract", "$trunc", "$and", "$or", "$not", "$cmp", "$eq", "$ne", "$gt", "$gte", "$lt",
    "$lte", "$cond", "$ifNull", "$switch", "$concat", "$substr", "$substrBytes", "$substrCP", "$toLower", "$toUpper",
    "$trim", "$ltrim", "$rtrim", "$split", "$strLenCP", "$strcasecmp", "$indexOfCP", "$regexMatch", "$regexFind",
    "$regexFindAll", "$replaceOne", "$replaceAll", "$arrayElemAt", "$arrayToObject", "$concatArrays", "$filter",
    "$in", "$indexOfArray", "$isArray", "$map", "$objectToArray", "$range", "$reduce", "$reverseArray", "$size",
    "$slice", "$zip", "$sortArray", "$dateAdd", "$dateSubtract", "$dateDiff", "$dateFromString", "$dateToString", "$dateTrunc",
    "$dayOfMonth", "$dayOfWeek", "$dayOfYear", "$hour", "$minute", "$month", "$second", "$week", "$year",
    "$isoWeek", "$millisecond", "$toBool", "$toDate", "$toDecimal", "$toDouble", "$toInt", "$toLong",
    "$toObjectId", "$toString", "$convert", "$type", "$literal", "$let", "$getField", "$setField", "$mergeObjects",
    "$meta", "$rand", "$setUnion", "$setIntersection", "$setDifference", "$setEquals", "$allElementsTrue",
    "$anyElementTrue", "$first", "$last", "$elemMatch", "$dateFromParts", "$dateToParts", "$isNumber",
}
# Chaves internas aceitas dentro de expressões ($cond: {if,then,else}, $switch: {branches, default}...).
EXPRESSION_ARGUMENT_KEYS = {
    "if", "then", "else", "branches", "case", "default", "input", "as", "in", "cond", "vars", "initialValue",
    "format", "date", "timezone", "onNull", "onError", "to", "startDate", "endDate", "unit", "amount",
    "binSize", "startOfWeek", "find", "replacement", "regex", "options", "chars", "sortBy", "n", "output",
    "field", "value", "limit", "dateString", "p", "method", "inputs", "useLongestLength", "defaults",
}

SEARCH_OPERATORS = {
    "text", "autocomplete", "compound", "range", "equals", "phrase", "wildcard", "regex", "exists", "in",
    "near", "moreLikeThis", "queryString", "geoWithin", "geoShape", "embeddedDocument", "span",
}
SEARCH_OPTIONS = {"index", "highlight", "count", "returnStoredSource", "scoreDetails", "sort", "concurrent",
                  "tracking", "facet", "searchAfter", "searchBefore", "returnScope"}
COMPOUND_CLAUSES = {"must", "mustNot", "should", "filter", "minimumShouldMatch", "score"}
SEARCH_OPERATOR_REQUIRED = {
    "text": {"query", "path"}, "autocomplete": {"query", "path"}, "phrase": {"query", "path"},
    "wildcard": {"query", "path"}, "regex": {"query", "path"}, "equals": {"path", "value"},
    "range": {"path"}, "exists": {"path"}, "in": {"path", "value"}, "near": {"path", "origin", "pivot"},
    "queryString": {"defaultPath", "query"}, "embeddedDocument": {"path", "operator"},
}
SEARCH_OPERATOR_KEYS = {
    "query", "path", "fuzzy", "score", "synonyms", "tokenOrder", "value", "gt", "gte", "lt", "lte",
    "allowAnalyzedField", "slop", "origin", "pivot", "defaultPath", "operator", "like", "matchCriteria",
}

# APIs reais do Console da IDE (ConsoleBootstrap.js). Não existem ConnectionPull nem GetDatabase.
CONSOLE_GLOBALS = {"db", "getConnection", "ConnectionPool", "ENV", "console", "EJSON", "UUID", "CGUUID", "JUUID",
                   "GUUID", "ObjectId", "NumberLong", "NumberDecimal", "ISODate"}
CONSOLE_CONNECTION_METHODS = {"getDatabase"}
CONSOLE_DATABASE_METHODS = {"getCollection", "getSiblingDB", "getName", "dropDatabase", "createCollection", "stats"}
CONSOLE_COLLECTION_METHODS = {
    "find", "aggregate", "findOne", "countDocuments", "estimatedDocumentCount", "distinct", "insertOne", "insertMany",
    "updateOne", "updateMany", "replaceOne", "deleteOne", "deleteMany", "drop", "createIndex", "dropIndex", "stats",
}
CONSOLE_CURSOR_METHODS = {"sort", "skip", "limit", "project", "hint", "collation", "comment", "batchSize", "maxTimeMS", "toArray"}

MONGOSH_GLOBALS = CONSOLE_GLOBALS - {"getConnection", "ConnectionPool", "ENV"} | {
    "print", "printjson", "use", "sleep", "Date", "Math", "JSON", "Object", "Array", "String", "Number",
    "Boolean", "RegExp", "Timestamp", "BinData", "MinKey", "MaxKey", "Decimal128", "Int32", "Long", "Double",
    "rs", "sh", "load", "Mongo", "connect", "Infinity", "NaN", "undefined", "parseInt", "parseFloat", "isNaN",
}
MONGOSH_DATABASE_METHODS = CONSOLE_DATABASE_METHODS | {
    "getCollectionNames", "listCollections", "runCommand", "adminCommand", "getMongo", "currentOp", "killOp",
    "serverStatus", "version", "hello", "createView", "aggregate", "getProfilingStatus", "setProfilingLevel",
}
MONGOSH_COLLECTION_METHODS = CONSOLE_COLLECTION_METHODS | {
    "bulkWrite", "findOneAndUpdate", "findOneAndReplace", "findOneAndDelete", "createIndexes", "getIndexes",
    "dropIndexes", "createSearchIndex", "getSearchIndexes", "updateSearchIndex", "dropSearchIndex", "explain",
    "watch", "renameCollection", "validate", "latencyStats", "getName", "count", "hideIndex", "unhideIndex",
}
MONGOSH_CURSOR_METHODS = CONSOLE_CURSOR_METHODS | {
    "forEach", "map", "hasNext", "next", "itcount", "pretty", "explain", "count", "size", "allowDiskUse",
    "readPref", "noCursorTimeout", "min", "max", "returnKey", "showRecordId", "tryNext", "close", "isExhausted",
}
EXPLAIN_METHODS = {"find", "aggregate", "count", "distinct", "remove", "update", "findAndModify"}

INDEX_KEY_VALUES = {1, -1, "text", "2dsphere", "2d", "hashed"}
INDEX_OPTIONS = {"name", "unique", "sparse", "partialFilterExpression", "expireAfterSeconds", "hidden", "collation",
                 "weights", "default_language", "language_override", "wildcardProjection", "background",
                 "2dsphereIndexVersion", "textIndexVersion"}
UPDATE_OPTIONS = {"upsert", "arrayFilters", "hint", "collation", "writeConcern", "let", "comment", "sort"}
CONSOLE_UPDATE_OPTIONS = {"upsert", "arrayFilters"}
FIND_AND_MODIFY_OPTIONS = UPDATE_OPTIONS | {"returnDocument", "projection", "returnNewDocument", "maxTimeMS"}

DIALECT_API = {
    "console": {
        "globals": CONSOLE_GLOBALS,
        "database": CONSOLE_DATABASE_METHODS,
        "collection": CONSOLE_COLLECTION_METHODS,
        "cursor": CONSOLE_CURSOR_METHODS,
    },
    "mongosh": {
        "globals": MONGOSH_GLOBALS,
        "database": MONGOSH_DATABASE_METHODS,
        "collection": MONGOSH_COLLECTION_METHODS,
        "cursor": MONGOSH_CURSOR_METHODS,
    },
}
