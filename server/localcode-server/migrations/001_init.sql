-- LocalCode PostgreSQL schema (production). Dev server uses in-memory store.

CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY,
    email TEXT UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'active'
);

CREATE TABLE IF NOT EXISTS api_tokens (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token_hash TEXT NOT NULL UNIQUE,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS device_codes (
    device_code TEXT PRIMARY KEY,
    user_code TEXT NOT NULL,
    user_id UUID REFERENCES users(id),
    expires_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS bench_suites (
    id TEXT PRIMARY KEY,
    slug TEXT NOT NULL,
    version TEXT NOT NULL,
    title TEXT NOT NULL,
    definition_json JSONB NOT NULL,
    publisher_id UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (slug, version)
);

CREATE TABLE IF NOT EXISTS bench_results (
    id UUID PRIMARY KEY,
    suite_id TEXT NOT NULL,
    suite_version TEXT NOT NULL,
    user_id UUID REFERENCES users(id),
    hf_model_id TEXT NOT NULL,
    quantization TEXT NOT NULL,
    backend TEXT NOT NULL,
    hardware_json JSONB NOT NULL,
    metrics_json JSONB NOT NULL,
    runner_version TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    visibility TEXT NOT NULL DEFAULT 'public'
);

CREATE INDEX IF NOT EXISTS idx_bench_results_model ON bench_results (hf_model_id);
CREATE INDEX IF NOT EXISTS idx_bench_results_quant ON bench_results (quantization);
CREATE INDEX IF NOT EXISTS idx_bench_results_suite ON bench_results (suite_id);
CREATE INDEX IF NOT EXISTS idx_bench_results_created ON bench_results (created_at DESC);

CREATE TABLE IF NOT EXISTS ledger_accounts (
    user_id UUID PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE,
    currency TEXT NOT NULL DEFAULT 'USDC',
    available NUMERIC(20, 8) NOT NULL DEFAULT 0,
    held NUMERIC(20, 8) NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS ledger_entries (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    type TEXT NOT NULL,
    amount NUMERIC(20, 8) NOT NULL,
    ref_type TEXT,
    ref_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata_json JSONB NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS deposits (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    chain TEXT NOT NULL,
    asset TEXT NOT NULL,
    address TEXT NOT NULL,
    txid TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    amount NUMERIC(20, 8),
    confirmations INT NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cloud_deployments (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id),
    provider TEXT NOT NULL,
    status TEXT NOT NULL,
    cost_hold NUMERIC(20, 8),
    endpoint TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
