//! LocalCode VPS API — in-memory store for dev; swap to Postgres via sqlx in production.

use axum::extract::{Path, Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use chrono::Utc;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tower_http::trace::TraceLayer;
use tracing::info;
use uuid::Uuid;

#[derive(Clone, Default)]
struct AppState {
    inner: Arc<Mutex<Db>>,
}

#[derive(Default)]
struct Db {
    users: HashMap<String, User>,
    tokens: HashMap<String, String>, // token_hash -> user_id
    device_codes: HashMap<String, DevicePending>,
    bench_results: Vec<BenchResultRow>,
    suites: Vec<SuiteRow>,
    balances: HashMap<String, Balance>,
    deposits: HashMap<String, Deposit>,
    ledger: Vec<LedgerEntry>,
    akash_deployments: HashMap<String, serde_json::Value>,
}

#[derive(Clone, Serialize)]
struct User {
    id: String,
    email: Option<String>,
    created_at: String,
    status: String,
}

#[derive(Clone)]
struct DevicePending {
    user_code: String,
    created_at: chrono::DateTime<Utc>,
    approved_user: Option<String>,
}

#[derive(Clone, Serialize, Deserialize)]
struct BenchResultRow {
    id: String,
    suite_id: String,
    suite_version: String,
    user_id: Option<String>,
    hf_model_id: String,
    quantization: String,
    backend: String,
    hardware_json: serde_json::Value,
    metrics_json: serde_json::Value,
    runner_version: String,
    created_at: String,
    visibility: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct SuiteRow {
    id: String,
    slug: String,
    version: String,
    title: String,
    definition_json: serde_json::Value,
    publisher_id: Option<String>,
    created_at: String,
}

#[derive(Clone, Serialize, Deserialize)]
struct Balance {
    currency: String,
    available: f64,
    held: f64,
}

#[derive(Clone, Serialize, Deserialize)]
struct Deposit {
    id: String,
    user_id: String,
    chain: String,
    asset: String,
    address: String,
    status: String,
    amount: Option<f64>,
}

#[derive(Clone, Serialize, Deserialize)]
struct LedgerEntry {
    id: String,
    user_id: String,
    entry_type: String,
    amount: f64,
    created_at: String,
    metadata: serde_json::Value,
}

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .init();

    let state = AppState::default();
    // Seed sample suite
    {
        let mut db = state.inner.lock().unwrap();
        db.suites.push(SuiteRow {
            id: "localcode-sample-coding".into(),
            slug: "localcode-sample-coding".into(),
            version: "1.0.0".into(),
            title: "Sample Coding Suite".into(),
            definition_json: serde_json::json!({"tasks": 2}),
            publisher_id: None,
            created_at: Utc::now().to_rfc3339(),
        });
    }

    let app = Router::new()
        .route("/v1/health", get(health))
        .route("/v1/client/min-version", get(min_version))
        .route("/v1/auth/device/start", post(device_start))
        .route("/v1/auth/device/poll", post(device_poll))
        .route("/v1/auth/logout", post(logout))
        .route("/v1/me", get(me))
        .route("/v1/models/trending", get(trending))
        .route("/v1/models/popular", get(popular))
        .route("/v1/bench/suites", get(list_suites).post(publish_suite))
        .route("/v1/bench/results", get(list_results).post(publish_result))
        .route("/v1/bench/results/{id}", get(get_result))
        .route("/v1/billing/balance", get(balance))
        .route("/v1/billing/deposits", post(create_deposit))
        .route("/v1/billing/deposits/{id}", get(get_deposit))
        .route(
            "/v1/billing/transactions",
            get(list_tx).post(create_tx),
        )
        .route("/v1/billing/quotes/akash", get(akash_quote))
        .route(
            "/v1/cloud/akash/deployments",
            post(akash_deploy),
        )
        .route(
            "/v1/cloud/akash/deployments/{id}",
            delete(akash_destroy),
        )
        // Dev-only: approve device code
        .route("/v1/auth/device/approve", post(device_approve))
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let addr: SocketAddr = std::env::var("LOCALCODE_SERVER_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8787".into())
        .parse()
        .expect("addr");
    info!("LocalCode server listening on {addr}");
    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

async fn min_version() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "min_version": "0.1.0" }))
}

#[derive(Serialize)]
struct DeviceStartResp {
    device_code: String,
    user_code: String,
    verification_url: String,
    interval: u64,
    expires_in: u64,
}

async fn device_start(State(state): State<AppState>) -> Json<DeviceStartResp> {
    let device_code = Uuid::new_v4().to_string();
    let user_code = format!("{:06}", rand::random::<u32>() % 1_000_000);
    let mut db = state.inner.lock().unwrap();
    db.device_codes.insert(
        device_code.clone(),
        DevicePending {
            user_code: user_code.clone(),
            created_at: Utc::now(),
            approved_user: None,
        },
    );
    Json(DeviceStartResp {
        device_code,
        user_code,
        verification_url: "https://localcode.example/device".into(),
        interval: 5,
        expires_in: 900,
    })
}

#[derive(Deserialize)]
struct DevicePollReq {
    device_code: String,
}

#[derive(Serialize)]
struct DevicePollResp {
    status: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    access_token: Option<String>,
}

async fn device_poll(
    State(state): State<AppState>,
    Json(req): Json<DevicePollReq>,
) -> Result<Json<DevicePollResp>, StatusCode> {
    let mut db = state.inner.lock().unwrap();
    let approved_user = db
        .device_codes
        .get(&req.device_code)
        .ok_or(StatusCode::NOT_FOUND)?
        .approved_user
        .clone();
    if let Some(user_id) = approved_user {
        let token = Uuid::new_v4().to_string();
        let hash = hash_token(&token);
        db.tokens.insert(hash, user_id);
        return Ok(Json(DevicePollResp {
            status: "approved".into(),
            access_token: Some(token),
        }));
    }
    // Auto-approve in dev if LOCALCODE_DEV_AUTO_AUTH=1
    if std::env::var("LOCALCODE_DEV_AUTO_AUTH").ok().as_deref() == Some("1") {
        let user_id = Uuid::new_v4().to_string();
        db.users.insert(
            user_id.clone(),
            User {
                id: user_id.clone(),
                email: Some("dev@localcode.example".into()),
                created_at: Utc::now().to_rfc3339(),
                status: "active".into(),
            },
        );
        db.balances.insert(
            user_id.clone(),
            Balance {
                currency: "USDC".into(),
                available: 10.0,
                held: 0.0,
            },
        );
        if let Some(p) = db.device_codes.get_mut(&req.device_code) {
            p.approved_user = Some(user_id.clone());
        }
        let token = Uuid::new_v4().to_string();
        db.tokens.insert(hash_token(&token), user_id);
        return Ok(Json(DevicePollResp {
            status: "approved".into(),
            access_token: Some(token),
        }));
    }
    Ok(Json(DevicePollResp {
        status: "pending".into(),
        access_token: None,
    }))
}

#[derive(Deserialize)]
struct ApproveReq {
    user_code: String,
    email: Option<String>,
}

async fn device_approve(
    State(state): State<AppState>,
    Json(req): Json<ApproveReq>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let mut db = state.inner.lock().unwrap();
    let user_id = Uuid::new_v4().to_string();
    db.users.insert(
        user_id.clone(),
        User {
            id: user_id.clone(),
            email: req.email,
            created_at: Utc::now().to_rfc3339(),
            status: "active".into(),
        },
    );
    db.balances.insert(
        user_id.clone(),
        Balance {
            currency: "USDC".into(),
            available: 0.0,
            held: 0.0,
        },
    );
    for p in db.device_codes.values_mut() {
        if p.user_code == req.user_code {
            p.approved_user = Some(user_id.clone());
            return Ok(Json(serde_json::json!({ "ok": true, "user_id": user_id })));
        }
    }
    Err(StatusCode::NOT_FOUND)
}

async fn logout(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> StatusCode {
    if let Some(token) = bearer(&headers) {
        let mut db = state.inner.lock().unwrap();
        db.tokens.remove(&hash_token(&token));
    }
    StatusCode::NO_CONTENT
}

async fn me(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<User>, StatusCode> {
    let user = require_user(&state, &headers)?;
    Ok(Json(user))
}

#[derive(Deserialize)]
struct TaskQuery {
    task: Option<String>,
}

async fn trending(Query(q): Query<TaskQuery>) -> Json<Vec<serde_json::Value>> {
    let _ = q.task;
    Json(curated_coding_models())
}

async fn popular(Query(q): Query<TaskQuery>) -> Json<Vec<serde_json::Value>> {
    let _ = q.task;
    Json(curated_coding_models())
}

fn curated_coding_models() -> Vec<serde_json::Value> {
    vec![
        serde_json::json!({"id": "Qwen/Qwen2.5-Coder-7B-Instruct", "tags": ["code"], "downloads": 1_000_000}),
        serde_json::json!({"id": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", "tags": ["code"], "downloads": 800_000}),
        serde_json::json!({"id": "codellama/CodeLlama-7b-Instruct-hf", "tags": ["code"], "downloads": 700_000}),
        serde_json::json!({"id": "bigcode/starcoder2-7b", "tags": ["code"], "downloads": 500_000}),
        serde_json::json!({"id": "microsoft/Phi-3-mini-4k-instruct", "tags": ["code"], "downloads": 400_000}),
    ]
}

async fn list_suites(State(state): State<AppState>) -> Json<Vec<SuiteRow>> {
    let db = state.inner.lock().unwrap();
    Json(db.suites.clone())
}

async fn publish_suite(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(suite): Json<SuiteRow>,
) -> Result<(StatusCode, Json<SuiteRow>), StatusCode> {
    let user = require_user(&state, &headers)?;
    let mut row = suite;
    row.publisher_id = Some(user.id);
    row.created_at = Utc::now().to_rfc3339();
    if row.id.is_empty() {
        row.id = Uuid::new_v4().to_string();
    }
    let mut db = state.inner.lock().unwrap();
    db.suites.push(row.clone());
    Ok((StatusCode::CREATED, Json(row)))
}

#[derive(Deserialize)]
struct ResultsQuery {
    hf_model_id: Option<String>,
    suite_id: Option<String>,
    quantization: Option<String>,
}

async fn list_results(
    State(state): State<AppState>,
    Query(q): Query<ResultsQuery>,
) -> Json<Vec<BenchResultRow>> {
    let db = state.inner.lock().unwrap();
    let mut out: Vec<_> = db.bench_results.clone();
    if let Some(m) = &q.hf_model_id {
        out.retain(|r| &r.hf_model_id == m);
    }
    if let Some(s) = &q.suite_id {
        out.retain(|r| &r.suite_id == s);
    }
    if let Some(quant) = &q.quantization {
        out.retain(|r| &r.quantization == quant);
    }
    Json(out)
}

#[derive(Deserialize)]
struct PublishBody {
    hf_model_id: String,
    quantization: String,
    weight_source: Option<String>,
    backend: String,
    backend_version: Option<String>,
    precision_notes: Option<String>,
    hardware: serde_json::Value,
    suite_id: String,
    suite_version: String,
    metrics: serde_json::Value,
    started_at: Option<String>,
    finished_at: Option<String>,
    runner_version: String,
}

async fn publish_result(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<PublishBody>,
) -> Result<(StatusCode, Json<BenchResultRow>), (StatusCode, Json<serde_json::Value>)> {
    if body.hf_model_id.is_empty() || body.quantization.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(serde_json::json!({
                "error": "hf_model_id and quantization required"
            })),
        ));
    }
    let user = require_user(&state, &headers).map_err(|s| {
        (
            s,
            Json(serde_json::json!({"error": "auth required to publish"})),
        )
    })?;
    let row = BenchResultRow {
        id: Uuid::new_v4().to_string(),
        suite_id: body.suite_id,
        suite_version: body.suite_version,
        user_id: Some(user.id),
        hf_model_id: body.hf_model_id,
        quantization: body.quantization,
        backend: body.backend,
        hardware_json: body.hardware,
        metrics_json: body.metrics,
        runner_version: body.runner_version,
        created_at: Utc::now().to_rfc3339(),
        visibility: "public".into(),
    };
    let mut db = state.inner.lock().unwrap();
    db.bench_results.push(row.clone());
    Ok((StatusCode::CREATED, Json(row)))
}

async fn get_result(
    State(state): State<AppState>,
    Path(id): Path<String>,
) -> Result<Json<BenchResultRow>, StatusCode> {
    let db = state.inner.lock().unwrap();
    db.bench_results
        .iter()
        .find(|r| r.id == id)
        .cloned()
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn balance(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Balance>, StatusCode> {
    let user = require_user(&state, &headers)?;
    let db = state.inner.lock().unwrap();
    Ok(Json(db.balances.get(&user.id).cloned().unwrap_or(Balance {
        currency: "USDC".into(),
        available: 0.0,
        held: 0.0,
    })))
}

#[derive(Deserialize)]
struct DepositReq {
    chain: Option<String>,
    asset: Option<String>,
    amount: Option<f64>,
}

async fn create_deposit(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<DepositReq>,
) -> Result<Json<Deposit>, StatusCode> {
    let user = require_user(&state, &headers)?;
    let id = Uuid::new_v4().to_string();
    // Dev deposit address (Base USDC)
    let deposit = Deposit {
        id: id.clone(),
        user_id: user.id,
        chain: req.chain.unwrap_or_else(|| "base".into()),
        asset: req.asset.unwrap_or_else(|| "USDC".into()),
        address: format!("0xDEPOSIT{}", &id[..8]),
        status: "pending".into(),
        amount: req.amount,
    };
    let mut db = state.inner.lock().unwrap();
    db.deposits.insert(id, deposit.clone());
    Ok(Json(deposit))
}

async fn get_deposit(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<Json<Deposit>, StatusCode> {
    let user = require_user(&state, &headers)?;
    let db = state.inner.lock().unwrap();
    let d = db.deposits.get(&id).ok_or(StatusCode::NOT_FOUND)?;
    if d.user_id != user.id {
        return Err(StatusCode::FORBIDDEN);
    }
    Ok(Json(d.clone()))
}

async fn list_tx(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Json<Vec<LedgerEntry>>, StatusCode> {
    let user = require_user(&state, &headers)?;
    let db = state.inner.lock().unwrap();
    Ok(Json(
        db.ledger
            .iter()
            .filter(|e| e.user_id == user.id)
            .cloned()
            .collect(),
    ))
}

#[derive(Deserialize)]
struct TxReq {
    #[serde(rename = "type")]
    entry_type: String,
    amount: f64,
    ref_type: Option<String>,
}

async fn create_tx(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(req): Json<TxReq>,
) -> Result<Json<LedgerEntry>, StatusCode> {
    let user = require_user(&state, &headers)?;
    let mut db = state.inner.lock().unwrap();
    let bal = db.balances.entry(user.id.clone()).or_insert(Balance {
        currency: "USDC".into(),
        available: 0.0,
        held: 0.0,
    });
    match req.entry_type.as_str() {
        "hold" => {
            if bal.available < req.amount {
                return Err(StatusCode::PAYMENT_REQUIRED);
            }
            bal.available -= req.amount;
            bal.held += req.amount;
        }
        "deposit" | "adjust" => {
            bal.available += req.amount;
        }
        "capture" => {
            bal.held = (bal.held - req.amount).max(0.0);
        }
        "refund" => {
            bal.held = (bal.held - req.amount).max(0.0);
            bal.available += req.amount;
        }
        _ => return Err(StatusCode::BAD_REQUEST),
    }
    let entry = LedgerEntry {
        id: Uuid::new_v4().to_string(),
        user_id: user.id,
        entry_type: req.entry_type,
        amount: req.amount,
        created_at: Utc::now().to_rfc3339(),
        metadata: serde_json::json!({ "ref_type": req.ref_type }),
    };
    db.ledger.push(entry.clone());
    Ok(Json(entry))
}

async fn akash_quote() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "usd_per_hour": 0.45,
        "usdc_per_hour": 0.45,
        "stale_after_secs": 60,
        "fetched_at": Utc::now().to_rfc3339(),
        "chain": "base",
        "asset": "USDC",
    }))
}

async fn akash_deploy(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, StatusCode> {
    let user = require_user(&state, &headers)?;
    let id = Uuid::new_v4().to_string();
    let dep = serde_json::json!({
        "id": id,
        "user_id": user.id,
        "status": "provisioning",
        "body": body,
        "custody": "server-mediated escrow — see Setup disclosure",
    });
    let mut db = state.inner.lock().unwrap();
    db.akash_deployments.insert(id.clone(), dep.clone());
    Ok(Json(dep))
}

async fn akash_destroy(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(id): Path<String>,
) -> Result<StatusCode, StatusCode> {
    let _user = require_user(&state, &headers)?;
    let mut db = state.inner.lock().unwrap();
    db.akash_deployments.remove(&id);
    Ok(StatusCode::NO_CONTENT)
}

fn bearer(headers: &HeaderMap) -> Option<String> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(|s| s.to_string())
}

fn hash_token(token: &str) -> String {
    let mut h = Sha256::new();
    h.update(token.as_bytes());
    hex::encode(h.finalize())
}

fn require_user(state: &AppState, headers: &HeaderMap) -> Result<User, StatusCode> {
    let token = bearer(headers).ok_or(StatusCode::UNAUTHORIZED)?;
    let db = state.inner.lock().unwrap();
    let user_id = db
        .tokens
        .get(&hash_token(&token))
        .ok_or(StatusCode::UNAUTHORIZED)?;
    db.users
        .get(user_id)
        .cloned()
        .ok_or(StatusCode::UNAUTHORIZED)
}
