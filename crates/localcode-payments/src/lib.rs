//! Payments: USDC top-up on Base (v1 default chain).

use localcode_api_client::ApiClient;
use localcode_core::error::{ErrorCode, LocalCodeError};
use serde::{Deserialize, Serialize};
use tracing::info;

/// v1 locked default: USDC on Base L2.
pub const V1_CHAIN: &str = "base";
pub const V1_ASSET: &str = "USDC";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Balance {
    pub currency: String,
    pub available: f64,
    pub held: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DepositIntent {
    pub id: String,
    pub chain: String,
    pub asset: String,
    pub address: String,
    pub amount_hint: Option<f64>,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LedgerEntry {
    pub id: String,
    pub entry_type: String,
    pub amount: f64,
    pub created_at: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AkashQuote {
    pub usd_per_hour: f64,
    pub usdc_per_hour: f64,
    pub stale_after_secs: u64,
    pub fetched_at: String,
}

pub struct PaymentsClient {
    api: ApiClient,
}

impl PaymentsClient {
    pub fn new(api: ApiClient) -> Self {
        Self { api }
    }

    pub async fn balance(&self) -> Result<Balance, LocalCodeError> {
        self.api.get_json("/v1/billing/balance").await
    }

    pub async fn create_deposit(&self, amount: Option<f64>) -> Result<DepositIntent, LocalCodeError> {
        info!(chain = V1_CHAIN, asset = V1_ASSET, "create deposit intent");
        let body = serde_json::json!({
            "chain": V1_CHAIN,
            "asset": V1_ASSET,
            "amount": amount,
        });
        self.api.post_json("/v1/billing/deposits", &body).await
    }

    pub async fn transactions(&self) -> Result<Vec<LedgerEntry>, LocalCodeError> {
        self.api.get_json("/v1/billing/transactions").await
    }

    pub async fn akash_quote(&self) -> Result<AkashQuote, LocalCodeError> {
        self.api.get_json("/v1/billing/quotes/akash").await
    }

    /// Place a hold for Akash deploy. Requires confirmation flag.
    pub async fn hold_for_deploy(
        &self,
        amount: f64,
        confirmed: bool,
    ) -> Result<LedgerEntry, LocalCodeError> {
        if !confirmed {
            return Err(LocalCodeError::new(
                ErrorCode::PaymentConfirmRequired,
                format!("Confirm hold of {amount:.2} USDC for Akash deploy"),
            )
            .with_hint("Review quote and confirm in UI")
            .with_cause("Spend confirmation required"));
        }
        let bal = self.balance().await?;
        if bal.available < amount {
            return Err(LocalCodeError::new(
                ErrorCode::InsufficientBalance,
                format!(
                    "Need {amount:.2} USDC, available {:.2}",
                    bal.available
                ),
            )
            .with_hint("Top up via Settings → Wallet (USDC on Base)")
            .with_cause("In-app balance too low"));
        }
        let body = serde_json::json!({
            "type": "hold",
            "amount": amount,
            "ref_type": "akash_deploy",
        });
        self.api.post_json("/v1/billing/transactions", &body).await
    }
}
