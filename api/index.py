"""
FinGuard AI — Transaction Risk API
Vercel Serverless / FastAPI backend.

The API uses the synthetic transaction dataset in data/transactions.json.
It is intentionally stateless and requires no database for the demo.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "transactions.json"

with DATA_FILE.open("r", encoding="utf-8") as f:
    TRANSACTIONS: list[dict] = json.load(f)

BY_UTR = {str(row.get("UTR_ID", "")).upper(): row for row in TRANSACTIONS}

app = FastAPI(
    title="FinGuard AI API",
    version="1.0.0",
    description="Transaction lookup, fraud search and risk statistics API.",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# Useful for local development and separate frontend hosting.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


def risk_level(score: float) -> str:
    if score < 30:
        return "LOW"
    if score < 70:
        return "MEDIUM"
    return "HIGH"


def enrich(row: dict) -> dict:
    result = dict(row)
    score = float(result.get("Risk_Score_0_100") or 0)
    result["Risk_Level"] = risk_level(score)
    return result


@app.get("/api")
def api_root():
    return {
        "name": "FinGuard AI API",
        "version": "1.0.0",
        "status": "online",
        "records": len(TRANSACTIONS),
        "docs": "/api/docs",
    }


@app.get("/api/health")
def health():
    return {"status": "healthy", "records_loaded": len(TRANSACTIONS)}


@app.get("/api/transaction/{utr}")
def get_transaction(utr: str):
    row = BY_UTR.get(utr.upper())
    if row is None:
        raise HTTPException(status_code=404, detail="UTR ID not found")
    return enrich(row)


@app.get("/api/search")
def search_transactions(
    label: Optional[str] = Query(None, description="Fraud or Genuine"),
    risk: Optional[str] = Query(None, description="LOW, MEDIUM or HIGH"),
    method: Optional[str] = Query(None, description="UPI or Card"),
    state: Optional[str] = None,
    city: Optional[str] = None,
    min_amount: Optional[float] = Query(None, ge=0),
    max_amount: Optional[float] = Query(None, ge=0),
    min_risk: Optional[float] = Query(None, ge=0, le=100),
    max_risk: Optional[float] = Query(None, ge=0, le=100),
    limit: int = Query(20, ge=1, le=100),
):
    results = TRANSACTIONS

    if label:
        wanted = label.strip().lower()
        results = [r for r in results if str(r.get("Label", "")).lower() == wanted]

    if risk:
        wanted = risk.strip().upper()
        results = [
            r for r in results
            if risk_level(float(r.get("Risk_Score_0_100") or 0)) == wanted
        ]

    if method:
        wanted = method.strip().lower()
        results = [
            r for r in results
            if str(r.get("Payment_Method", "")).lower() == wanted
        ]

    if state:
        wanted = state.strip().lower()
        results = [r for r in results if str(r.get("State", "")).lower() == wanted]

    if city:
        wanted = city.strip().lower()
        results = [r for r in results if str(r.get("City", "")).lower() == wanted]

    if min_amount is not None:
        results = [r for r in results if float(r.get("Amount_INR") or 0) >= min_amount]

    if max_amount is not None:
        results = [r for r in results if float(r.get("Amount_INR") or 0) <= max_amount]

    if min_risk is not None:
        results = [
            r for r in results
            if float(r.get("Risk_Score_0_100") or 0) >= min_risk
        ]

    if max_risk is not None:
        results = [
            r for r in results
            if float(r.get("Risk_Score_0_100") or 0) <= max_risk
        ]

    return {
        "count": len(results),
        "returned": min(len(results), limit),
        "filters": {
            "label": label,
            "risk": risk,
            "method": method,
            "state": state,
            "city": city,
            "min_amount": min_amount,
            "max_amount": max_amount,
            "min_risk": min_risk,
            "max_risk": max_risk,
        },
        "results": [enrich(r) for r in results[:limit]],
    }


@app.get("/api/stats")
def stats():
    fraud = sum(1 for r in TRANSACTIONS if r.get("Label") == "Fraud")
    genuine = len(TRANSACTIONS) - fraud
    low = sum(
        1 for r in TRANSACTIONS
        if risk_level(float(r.get("Risk_Score_0_100") or 0)) == "LOW"
    )
    medium = sum(
        1 for r in TRANSACTIONS
        if risk_level(float(r.get("Risk_Score_0_100") or 0)) == "MEDIUM"
    )
    high = sum(
        1 for r in TRANSACTIONS
        if risk_level(float(r.get("Risk_Score_0_100") or 0)) == "HIGH"
    )
    return {
        "total": len(TRANSACTIONS),
        "fraud": fraud,
        "genuine": genuine,
        "risk_levels": {"low": low, "medium": medium, "high": high},
    }


def to_alert(row: dict) -> dict:
    score = float(row.get("Risk_Score_0_100") or 0)
    return {
        "alert_id": f"ALT-{row.get('UTR_ID', '')}",
        "utr_id": row.get("UTR_ID"),
        "transaction_id": row.get("Transaction_ID"),
        "timestamp": row.get("Transaction_Timestamp"),
        "amount_inr": row.get("Amount_INR"),
        "payment_method": row.get("Payment_Method"),
        "city": row.get("City"),
        "state": row.get("State"),
        "risk_score": score,
        "risk_level": risk_level(score),
        "label": row.get("Label"),
        "reason": row.get("Fraud_Risk_Reason"),
        "decision": "BLOCK" if score >= 85 else "FLAG",
    }


@app.get("/api/alerts")
def list_alerts(
    min_risk: float = Query(70, ge=0, le=100, description="Lowest risk score to include"),
    payment_method: Optional[str] = Query(None, description="UPI or Card"),
    state: Optional[str] = None,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Paginated, most-urgent-first feed of transactions that crossed the
    review threshold — the working queue behind the Allow/Flag/Block
    decision step."""
    flagged = [
        r for r in TRANSACTIONS
        if float(r.get("Risk_Score_0_100") or 0) >= min_risk
    ]

    if payment_method:
        wanted = payment_method.strip().lower()
        flagged = [r for r in flagged if str(r.get("Payment_Method", "")).lower() == wanted]

    if state:
        wanted = state.strip().lower()
        flagged = [r for r in flagged if str(r.get("State", "")).lower() == wanted]

    flagged.sort(
        key=lambda r: (float(r.get("Risk_Score_0_100") or 0), r.get("Transaction_Timestamp") or ""),
        reverse=True,
    )

    page = flagged[offset: offset + limit]

    return {
        "total_matching": len(flagged),
        "returned": len(page),
        "offset": offset,
        "limit": limit,
        "filters": {"min_risk": min_risk, "payment_method": payment_method, "state": state},
        "alerts": [to_alert(r) for r in page],
    }


@app.get("/api/alerts/reasons")
def alert_reasons(
    min_risk: float = Query(70, ge=0, le=100),
    top: int = Query(10, ge=1, le=50),
):
    """Aggregated counts of Fraud_Risk_Reason among flagged transactions —
    powers a 'top reasons we're flagging transactions' chart."""
    flagged = [
        r for r in TRANSACTIONS
        if float(r.get("Risk_Score_0_100") or 0) >= min_risk
    ]
    counts = Counter(
        r.get("Fraud_Risk_Reason") or "Unspecified" for r in flagged
    )
    ranked = counts.most_common(top)

    return {
        "min_risk": min_risk,
        "flagged_total": len(flagged),
        "reasons": [{"reason": reason, "count": count} for reason, count in ranked],
    }


@app.get("/api/alerts/{utr}")
def get_alert(utr: str):
    """Single alert detail, looked up by UTR ID."""
    row = BY_UTR.get(utr.upper())
    if row is None:
        raise HTTPException(status_code=404, detail="UTR ID not found")
    if float(row.get("Risk_Score_0_100") or 0) < 30:
        raise HTTPException(status_code=404, detail="This transaction was never flagged")
    return to_alert(row)


# Local development:
#   uvicorn api.index:app --reload
