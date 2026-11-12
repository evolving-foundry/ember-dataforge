/**
 * Runtime Protocol — dataset references (Phase 0 freeze).
 *
 * DatasetRef is the core cross-language boundary (doc §12/§14): TypeScript
 * only ever holds references; the actual Ray Dataset / Arrow / Parquet data
 * lives in the Python data plane behind the DatasetRegistry.
 */

export type DatasetBackend = "ray" | "parquet" | "arrow" | "object-store";

export interface DatasetSchemaField {
  name: string;
  type: string;
  /** Modality this column carries, when applicable: "text" | "image" | ... */
  modality?: string;
}

export interface DatasetSchema {
  fields: DatasetSchemaField[];
}

export interface DatasetMetadata {
  /** Number of rows, when known. */
  numRows?: number;
  /** Approximate size in bytes, when known. */
  sizeBytes?: number;
  /** Modalities present in the dataset. */
  modalities?: string[];
  /** Data plane location (path / object store URI), when materialized. */
  location?: string;
  /** Backend-specific extras (JSON-serializable). */
  extra?: Record<string, unknown>;
}

export interface DatasetRef {
  id: string;
  runtimeId: string;
  backend: DatasetBackend;
  schema?: DatasetSchema;
  metadata?: DatasetMetadata;
}
