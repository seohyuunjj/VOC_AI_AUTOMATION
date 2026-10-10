#!/usr/bin/env python3
"""Reproducible VOC pipeline for the Mini-BAR public app-review dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sqlite3
import statistics
import sys
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "config.json"
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
STREAM_DIR = ROOT / "data" / "stream"
RESULTS_DIR = ROOT / "results"
LABELS = ("problem_report", "feature_request", "irrelevant")
SOURCE_BASE = (
    "https://raw.githubusercontent.com/Jl-wei/mini-bar/"
    "bb6a7cfa064ebb4afbdd3853e7dbe6e31e0d3649/"
    "dataset/for_classification/"
)
SOURCE_FILES = (
    "Garmin Connect.en.csv",
    "Huawei Health.en.csv",
    "Samsung Health.en.csv",
)


@dataclass
class Prediction:
    review_id: str
    labels: list[str]
    summary: str
    severity: int
    confidence: float
    reason: str
    method: str


@dataclass
class Usage:
    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "Usage") -> None:
        self.requests += other.requests
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens

    def snapshot(self) -> "Usage":
        return Usage(**asdict(self))

    def minus(self, earlier: "Usage") -> "Usage":
        return Usage(
            requests=self.requests - earlier.requests,
            input_tokens=self.input_tokens - earlier.input_tokens,
            output_tokens=self.output_tokens - earlier.output_tokens,
            total_tokens=self.total_tokens - earlier.total_tokens,
        )


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def load_local_env() -> dict[str, str]:
    values: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def get_secret(*names: str) -> str | None:
    local_values = load_local_env()
    for name in names:
        value = os.environ.get(name) or local_values.get(name)
        if value:
            return value
    return None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = re.sub(r"https?://\S+", "<URL>", text)
    text = re.sub(r"\S+@\S+\.\S+", "<EMAIL>", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def content_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def source_review_id(app: str, source_row: int) -> str:
    value = f"mini-bar|{app}|{source_row}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:20]


def parse_gold(row: dict) -> list[str]:
    labels = [label for label in LABELS if str(row.get(label, "0")) == "1"]
    return labels or ["irrelevant"]


def download_data(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for filename in SOURCE_FILES:
        destination = RAW_DIR / filename
        if destination.exists() and not force:
            print(f"skip existing: {destination.relative_to(ROOT)}")
            continue
        url = SOURCE_BASE + urllib.parse.quote(filename)
        print(f"download: {filename}")
        request = urllib.request.Request(url, headers={"User-Agent": "voc-pipeline/1.0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read()
        destination.write_bytes(data)
        print(f"saved: {destination.relative_to(ROOT)} ({len(data):,} bytes)")


def write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def prepare_data() -> None:
    missing = [name for name in SOURCE_FILES if not (RAW_DIR / name).exists()]
    if missing:
        raise SystemExit(f"Missing source files: {missing}. Run `python3 voc_pipeline.py download`.")

    rows: list[dict] = []
    for filename in SOURCE_FILES:
        source_rows = read_csv(RAW_DIR / filename)
        for index, row in enumerate(source_rows, start=1):
            text = normalize_text(row["data"])
            if not text:
                continue
            app = row["app"].strip()
            rows.append(
                {
                    "review_id": source_review_id(app, index),
                    "app": app,
                    "source_row": index,
                    "review_text": row["data"].strip(),
                    "normalized_text": text,
                    "content_hash": content_hash(text),
                    "score": int(row["score"]),
                    "thumbs_up_count": int(row.get("thumbsUpCount", 0) or 0),
                    "problem_report": int(row["problem_report"]),
                    "feature_request": int(row["feature_request"]),
                    "irrelevant": int(row["irrelevant"]),
                    "gold_labels": "|".join(parse_gold(row)),
                    "source_file": filename,
                }
            )

    fieldnames = list(rows[0])
    all_path = PROCESSED_DIR / "all_reviews.csv"
    write_csv(all_path, rows, fieldnames)

    config = load_config()
    shuffled = rows.copy()
    random.Random(config["random_seed"]).shuffle(shuffled)
    day_count = int(config["stream_days"])
    for day in range(1, day_count + 1):
        batch = shuffled[day - 1 :: day_count]
        write_csv(STREAM_DIR / f"day_{day}.csv", batch, fieldnames)

    stats = {
        "prepared_at": utc_now(),
        "rows": len(rows),
        "apps": sorted({row["app"] for row in rows}),
        "label_counts": {
            label: sum(int(row[label]) for row in rows) for label in LABELS
        },
        "source_commit": "bb6a7cfa064ebb4afbdd3853e7dbe6e31e0d3649",
        "stream_days": day_count,
        "batch_sizes": {
            f"day_{day}": len(read_csv(STREAM_DIR / f"day_{day}.csv"))
            for day in range(1, day_count + 1)
        },
    }
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (PROCESSED_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))


PROBLEM_RE = re.compile(
    r"\b(crash(?:es|ed|ing)?|unable to|cannot|can't|doesn't work|does not work|"
    r"not working|stopped working|error(?:s)?|won't (?:open|sync|connect|start)|"
    r"fail(?:ed|s|ing) to|freeze(?:s|d)?|frozen|doesn't (?:sync|connect|open|start)|"
    r"not (?:syncing|connecting|opening))\b",
    re.IGNORECASE,
)
FEATURE_RE = re.compile(
    r"\b(please add|would like (?:an?|the|to)|wish (?:it|there)|"
    r"should (?:add|have|allow)|need (?:an? |the )?(?:option|feature)|"
    r"feature request|missing feature|bring back|add (?:an?|the) |"
    r"please (?:make|allow|include))\b",
    re.IGNORECASE,
)


def rule_classify(row: dict) -> Prediction | None:
    text = row["normalized_text"]
    has_problem = bool(PROBLEM_RE.search(text))
    has_feature = bool(FEATURE_RE.search(text))
    word_count = len(text.split())
    score = int(row.get("score", 3))

    if has_problem and not has_feature:
        labels = ["problem_report"]
        reason = "명시적인 장애·작동 실패 표현"
        confidence = 0.86
    elif has_feature and not has_problem:
        labels = ["feature_request"]
        reason = "명시적인 기능 추가·변경 요청 표현"
        confidence = 0.88
    elif word_count <= 8 and not has_problem and not has_feature:
        labels = ["irrelevant"]
        reason = "요구사항 단서가 없는 8단어 이하의 짧은 평가"
        confidence = 0.82
    else:
        return None

    severity = 4 if labels == ["problem_report"] and score <= 2 else 2
    return Prediction(
        review_id=row["review_id"],
        labels=labels,
        summary=text[:120],
        severity=severity,
        confidence=confidence,
        reason=reason,
        method="rule",
    )


class OracleProvider:
    """Offline test provider. It must never be used for reported AI results."""

    def __init__(self, _config: dict):
        self.usage = Usage()

    def classify(self, rows: Sequence[dict]) -> list[Prediction]:
        return [
            Prediction(
                review_id=row["review_id"],
                labels=row["gold_labels"].split("|"),
                summary=row["normalized_text"][:120],
                severity=4 if int(row["score"]) <= 2 else 2,
                confidence=1.0,
                reason="오프라인 파이프라인 검증용 정답 제공자",
                method="oracle",
            )
            for row in rows
        ]


class OpenAIProvider:
    def __init__(self, config: dict, model: str | None = None):
        api_key = get_secret("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY is not set.")
        self.api_key = api_key
        self.model = model or config["model"]
        self.batch_size = int(config["batch_size"])
        self.timeout = int(config["request_timeout_seconds"])
        self.max_retries = int(config["max_retries"])
        self.usage = Usage()

    @staticmethod
    def schema() -> dict:
        item = {
            "type": "object",
            "properties": {
                "review_id": {"type": "string"},
                "labels": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(LABELS)},
                },
                "summary": {"type": "string"},
                "severity": {"type": "integer", "minimum": 1, "maximum": 5},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "reason": {"type": "string"},
            },
            "required": [
                "review_id",
                "labels",
                "summary",
                "severity",
                "confidence",
                "reason",
            ],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {"classifications": {"type": "array", "items": item}},
            "required": ["classifications"],
            "additionalProperties": False,
        }

    def _request(self, rows: Sequence[dict]) -> list[Prediction]:
        reviews = [
            {
                "review_id": row["review_id"],
                "app": row["app"],
                "score": int(row["score"]),
                "review": row["normalized_text"],
            }
            for row in rows
        ]
        system = (
            "You classify English mobile-app reviews for VOC analysis. "
            "Return one or more labels: problem_report for bugs or failures; "
            "feature_request for requested additions or changes; irrelevant for "
            "ratings or comments that are neither. Use irrelevant only when the "
            "other two do not apply. Keep summaries and reasons concise."
        )
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": "Classify this JSON array:\n" + json.dumps(reviews, ensure_ascii=False),
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "voc_batch_classification",
                    "strict": True,
                    "schema": self.schema(),
                }
            },
        }
        request = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"OpenAI HTTP {error.code}: {detail}")
                if "insufficient_quota" in detail or "credit_balance_exhausted" in detail:
                    raise RuntimeError(
                        "OpenAI API credit is exhausted. Add API credits, then rerun the command."
                    ) from last_error
                if error.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise last_error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            time.sleep(min(2**attempt, 8))
        else:
            raise RuntimeError(f"OpenAI request failed after retries: {last_error}")

        usage = body.get("usage") or {}
        self.usage.add(
            Usage(
                requests=1,
                input_tokens=int(usage.get("input_tokens", 0)),
                output_tokens=int(usage.get("output_tokens", 0)),
                total_tokens=int(usage.get("total_tokens", 0)),
            )
        )

        output_text = ""
        for output in body.get("output", []):
            if output.get("type") != "message":
                continue
            for content in output.get("content", []):
                if content.get("type") == "output_text":
                    output_text += content.get("text", "")
                elif content.get("type") == "refusal":
                    raise RuntimeError(f"Model refusal: {content.get('refusal', '')}")
        if not output_text:
            raise RuntimeError(f"No output_text in response status={body.get('status')}")

        parsed = json.loads(output_text)
        expected_ids = {row["review_id"] for row in rows}
        predictions: list[Prediction] = []
        for item in parsed["classifications"]:
            labels = list(dict.fromkeys(item["labels"])) or ["irrelevant"]
            if item["review_id"] not in expected_ids:
                raise RuntimeError(f"Unexpected review_id: {item['review_id']}")
            predictions.append(
                Prediction(
                    review_id=item["review_id"],
                    labels=labels,
                    summary=item["summary"],
                    severity=item["severity"],
                    confidence=item["confidence"],
                    reason=item["reason"],
                    method="openai",
                )
            )
        received_ids = {prediction.review_id for prediction in predictions}
        if received_ids != expected_ids:
            missing = expected_ids - received_ids
            raise RuntimeError(f"Missing classifications: {sorted(missing)}")
        return predictions

    def classify(self, rows: Sequence[dict]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            predictions.extend(self._request(batch))
        return predictions


class GeminiProvider:
    """Gemini Developer API adapter using REST and structured JSON output."""

    def __init__(self, config: dict, model: str | None = None):
        api_key = get_secret("GEMINI_API_KEY", "GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit(
                "GEMINI_API_KEY is not set. Create a free key in Google AI Studio "
                "and export it before running this provider."
            )
        self.api_key = api_key
        self.model = model or config["gemini_model"]
        self.batch_size = int(config["gemini_batch_size"])
        self.min_interval = float(config["gemini_min_interval_seconds"])
        self.timeout = int(config["request_timeout_seconds"])
        self.max_retries = int(config["max_retries"])
        self.usage = Usage()
        self._last_request_at = 0.0

    def _pace(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if self._last_request_at and elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)

    def _request(self, rows: Sequence[dict]) -> list[Prediction]:
        reviews = [
            {
                "review_id": row["review_id"],
                "app": row["app"],
                "score": int(row["score"]),
                "review": row["normalized_text"],
            }
            for row in rows
        ]
        system = (
            "You classify English mobile-app reviews for VOC analysis. "
            "Return one or more labels: problem_report for bugs or failures; "
            "feature_request for requested additions or changes; irrelevant for "
            "ratings or comments that are neither. Use irrelevant only when the "
            "other two do not apply. Keep summaries and reasons concise."
        )
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "text": "Classify this JSON array:\n"
                            + json.dumps(reviews, ensure_ascii=False)
                        }
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseJsonSchema": OpenAIProvider.schema(),
            },
        }
        encoded_model = urllib.parse.quote(self.model, safe="")
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{encoded_model}:generateContent"
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-goog-api-key": self.api_key,
                "Content-Type": "application/json",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            self._pace()
            self._last_request_at = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"Gemini HTTP {error.code}: {detail}")
                if error.code not in {408, 429, 500, 502, 503, 504}:
                    raise last_error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            time.sleep(min(2**attempt, 8))
        else:
            raise RuntimeError(f"Gemini request failed after retries: {last_error}")

        usage = body.get("usageMetadata") or {}
        prompt_tokens = int(usage.get("promptTokenCount", 0))
        candidate_tokens = int(usage.get("candidatesTokenCount", 0))
        thought_tokens = int(usage.get("thoughtsTokenCount", 0))
        self.usage.add(
            Usage(
                requests=1,
                input_tokens=prompt_tokens,
                output_tokens=candidate_tokens + thought_tokens,
                total_tokens=int(
                    usage.get(
                        "totalTokenCount",
                        prompt_tokens + candidate_tokens + thought_tokens,
                    )
                ),
            )
        )

        candidates = body.get("candidates") or []
        if not candidates:
            reason = (body.get("promptFeedback") or {}).get("blockReason", "unknown")
            raise RuntimeError(f"Gemini returned no candidate. blockReason={reason}")
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason")
        if finish_reason not in {None, "STOP"}:
            raise RuntimeError(f"Gemini response did not complete: {finish_reason}")
        output_text = "".join(
            part.get("text", "")
            for part in (candidate.get("content") or {}).get("parts", [])
            if "text" in part
        )
        if not output_text:
            raise RuntimeError("Gemini response contained no text output.")

        parsed = json.loads(output_text)
        expected_ids = {row["review_id"] for row in rows}
        predictions: list[Prediction] = []
        for item in parsed["classifications"]:
            labels = list(dict.fromkeys(item["labels"])) or ["irrelevant"]
            if item["review_id"] not in expected_ids:
                raise RuntimeError(f"Unexpected review_id: {item['review_id']}")
            predictions.append(
                Prediction(
                    review_id=item["review_id"],
                    labels=labels,
                    summary=item["summary"],
                    severity=item["severity"],
                    confidence=item["confidence"],
                    reason=item["reason"],
                    method="gemini",
                )
            )
        received_ids = {prediction.review_id for prediction in predictions}
        if received_ids != expected_ids:
            missing = expected_ids - received_ids
            raise RuntimeError(f"Missing classifications: {sorted(missing)}")
        return predictions

    def classify(self, rows: Sequence[dict]) -> list[Prediction]:
        predictions: list[Prediction] = []
        for start in range(0, len(rows), self.batch_size):
            batch = rows[start : start + self.batch_size]
            predictions.extend(self._request(batch))
        return predictions


def resolved_model(provider_name: str, config: dict, model: str | None) -> str:
    if model:
        return model
    if provider_name == "openai":
        return config["model"]
    if provider_name == "gemini":
        return config["gemini_model"]
    return "oracle"


def make_provider(name: str, config: dict, model: str | None = None):
    if name == "oracle":
        return OracleProvider(config)
    if name == "openai":
        return OpenAIProvider(config, model=model)
    if name == "gemini":
        return GeminiProvider(config, model=model)
    raise ValueError(name)


def classify_rows(rows: Sequence[dict], mode: str, provider) -> tuple[list[Prediction], int]:
    if mode == "baseline":
        return provider.classify(rows), len(rows)

    predictions: list[Prediction] = []
    ai_rows: list[dict] = []
    for row in rows:
        prediction = rule_classify(row)
        if prediction is None:
            ai_rows.append(row)
        else:
            predictions.append(prediction)
    predictions.extend(provider.classify(ai_rows))
    order = {row["review_id"]: index for index, row in enumerate(rows)}
    predictions.sort(key=lambda item: order[item.review_id])
    return predictions, len(ai_rows)


def multilabel_metrics(rows: Sequence[dict], predictions: Sequence[Prediction]) -> dict:
    pred_by_id = {prediction.review_id: set(prediction.labels) for prediction in predictions}
    exact = 0
    per_label = {}
    total_tp = total_fp = total_fn = 0
    for label in LABELS:
        tp = fp = fn = 0
        for row in rows:
            gold = set(row["gold_labels"].split("|"))
            pred = pred_by_id[row["review_id"]]
            tp += int(label in gold and label in pred)
            fp += int(label not in gold and label in pred)
            fn += int(label in gold and label not in pred)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1}
        total_tp += tp
        total_fp += fp
        total_fn += fn
    for row in rows:
        gold = set(row["gold_labels"].split("|"))
        exact += int(gold == pred_by_id[row["review_id"]])
    micro_precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if micro_precision + micro_recall
        else 0.0
    )
    return {
        "exact_match": exact / len(rows),
        "micro_f1": micro_f1,
        "macro_f1": statistics.mean(item["f1"] for item in per_label.values()),
        "per_label": per_label,
    }


def select_sample(rows: list[dict], sample_size: int, seed: int) -> list[dict]:
    if sample_size > len(rows):
        raise SystemExit(f"sample-size {sample_size} exceeds dataset size {len(rows)}")
    selected = random.Random(seed).sample(rows, sample_size)
    return selected


def benchmark(provider_name: str, sample_size: int, model: str | None) -> Path:
    dataset_path = PROCESSED_DIR / "all_reviews.csv"
    if not dataset_path.exists():
        raise SystemExit("Prepared dataset not found. Run `download` and `prepare` first.")
    config = load_config()
    rows = select_sample(read_csv(dataset_path), sample_size, int(config["random_seed"]))
    day_count = int(config["stream_days"])
    day_batches = [rows[day - 1 :: day_count] for day in range(1, day_count + 1)]

    result: dict = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "created_at": utc_now(),
        "provider": provider_name,
        "model": resolved_model(provider_name, config, model),
        "sample_size": sample_size,
        "stream_days": day_count,
        "seed": config["random_seed"],
        "modes": {},
    }
    prediction_rows: list[dict] = []

    for mode in ("baseline", "hybrid"):
        provider = make_provider(provider_name, config, model=model)
        all_predictions: list[Prediction] = []
        daily: list[dict] = []
        started = time.perf_counter()
        total_ai_reviews = 0
        for day, batch in enumerate(day_batches, start=1):
            before = provider.usage.snapshot()
            day_started = time.perf_counter()
            predictions, ai_reviews = classify_rows(batch, mode, provider)
            elapsed = time.perf_counter() - day_started
            day_usage = provider.usage.minus(before)
            all_predictions.extend(predictions)
            total_ai_reviews += ai_reviews
            daily.append(
                {
                    "day": day,
                    "reviews": len(batch),
                    "ai_reviews": ai_reviews,
                    "elapsed_seconds": elapsed,
                    "usage": asdict(day_usage),
                }
            )
            print(
                f"{mode} day={day} reviews={len(batch)} ai={ai_reviews} "
                f"seconds={elapsed:.2f} tokens={day_usage.total_tokens}"
            )
        elapsed_total = time.perf_counter() - started
        metrics = multilabel_metrics(rows, all_predictions)
        result["modes"][mode] = {
            "elapsed_seconds": elapsed_total,
            "ai_reviews": total_ai_reviews,
            "rule_reviews": sample_size - total_ai_reviews,
            "usage": asdict(provider.usage),
            "metrics": metrics,
            "daily": daily,
        }
        by_id = {row["review_id"]: row for row in rows}
        for prediction in all_predictions:
            source = by_id[prediction.review_id]
            prediction_rows.append(
                {
                    "mode": mode,
                    "review_id": prediction.review_id,
                    "app": source["app"],
                    "score": source["score"],
                    "gold_labels": source["gold_labels"],
                    "predicted_labels": "|".join(prediction.labels),
                    "method": prediction.method,
                    "confidence": prediction.confidence,
                    "review_text": source["normalized_text"],
                }
            )

    baseline = result["modes"]["baseline"]
    hybrid = result["modes"]["hybrid"]
    base_tokens = baseline["usage"]["total_tokens"]
    base_time = baseline["elapsed_seconds"]
    result["comparison"] = {
        "time_reduction_percent": (
            (base_time - hybrid["elapsed_seconds"]) / base_time * 100 if base_time else 0.0
        ),
        "token_reduction_percent": (
            (base_tokens - hybrid["usage"]["total_tokens"]) / base_tokens * 100
            if base_tokens
            else 0.0
        ),
        "ai_call_review_reduction_percent": (
            (baseline["ai_reviews"] - hybrid["ai_reviews"])
            / baseline["ai_reviews"]
            * 100
            if baseline["ai_reviews"]
            else 0.0
        ),
        "hybrid_minus_baseline_micro_f1": (
            hybrid["metrics"]["micro_f1"] - baseline["metrics"]["micro_f1"]
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{result['run_id']}_{provider_name}"
    result_path = RESULTS_DIR / f"benchmark_{suffix}.json"
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(
        RESULTS_DIR / f"benchmark_predictions_{suffix}.csv",
        prediction_rows,
        list(prediction_rows[0]),
    )
    print(json.dumps(result["comparison"], ensure_ascii=False, indent=2))
    print(f"saved: {result_path.relative_to(ROOT)}")
    return result_path


def init_db(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY,
            app TEXT NOT NULL,
            review_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            score INTEGER NOT NULL,
            thumbs_up_count INTEGER NOT NULL,
            gold_labels TEXT NOT NULL,
            first_seen_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS analyses (
            review_id TEXT PRIMARY KEY,
            labels TEXT NOT NULL,
            summary TEXT NOT NULL,
            severity INTEGER NOT NULL,
            confidence REAL NOT NULL,
            reason TEXT NOT NULL,
            method TEXT NOT NULL,
            analyzed_at TEXT NOT NULL,
            FOREIGN KEY (review_id) REFERENCES reviews(review_id)
        );
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            day INTEGER NOT NULL,
            dry_run INTEGER NOT NULL,
            provider TEXT NOT NULL,
            input_rows INTEGER NOT NULL,
            new_rows INTEGER NOT NULL,
            changed_rows INTEGER NOT NULL,
            unchanged_rows INTEGER NOT NULL,
            ai_rows INTEGER NOT NULL,
            elapsed_seconds REAL NOT NULL,
            total_tokens INTEGER NOT NULL
        );
        """
    )


def ingest(day: int, dry_run: bool, provider_name: str, model: str | None) -> Path:
    batch_path = STREAM_DIR / f"day_{day}.csv"
    if not batch_path.exists():
        raise SystemExit(f"Batch not found: {batch_path}. Run `prepare` first.")
    config = load_config()
    rows = read_csv(batch_path)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(RESULTS_DIR / "voc.db")
    init_db(connection)

    new_rows: list[dict] = []
    changed_rows: list[dict] = []
    unchanged = 0
    for row in rows:
        existing = connection.execute(
            "SELECT content_hash FROM reviews WHERE review_id = ?", (row["review_id"],)
        ).fetchone()
        if existing is None:
            new_rows.append(row)
        elif existing[0] != row["content_hash"]:
            changed_rows.append(row)
        else:
            unchanged += 1

    targets = new_rows + changed_rows
    provider = make_provider(provider_name, config, model=model)
    started = time.perf_counter()
    predictions, ai_rows = classify_rows(targets, "hybrid", provider) if targets else ([], 0)
    elapsed = time.perf_counter() - started
    prediction_by_id = {prediction.review_id: prediction for prediction in predictions}
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    report = {
        "run_id": run_id,
        "created_at": utc_now(),
        "day": day,
        "dry_run": dry_run,
        "provider": provider_name,
        "model": resolved_model(provider_name, config, model),
        "counts": {
            "input": len(rows),
            "new": len(new_rows),
            "changed": len(changed_rows),
            "unchanged": unchanged,
            "rule": len(targets) - ai_rows,
            "ai": ai_rows,
        },
        "elapsed_seconds": elapsed,
        "usage": asdict(provider.usage),
        "preview": [asdict(prediction) for prediction in predictions[:25]],
    }

    if not dry_run:
        now = utc_now()
        for row in targets:
            connection.execute(
                """
                INSERT INTO reviews (
                    review_id, app, review_text, normalized_text, content_hash, score,
                    thumbs_up_count, gold_labels, first_seen_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    app=excluded.app, review_text=excluded.review_text,
                    normalized_text=excluded.normalized_text,
                    content_hash=excluded.content_hash, score=excluded.score,
                    thumbs_up_count=excluded.thumbs_up_count,
                    gold_labels=excluded.gold_labels, updated_at=excluded.updated_at
                """,
                (
                    row["review_id"], row["app"], row["review_text"],
                    row["normalized_text"], row["content_hash"], int(row["score"]),
                    int(row["thumbs_up_count"]), row["gold_labels"], now, now,
                ),
            )
            prediction = prediction_by_id[row["review_id"]]
            connection.execute(
                """
                INSERT INTO analyses (
                    review_id, labels, summary, severity, confidence, reason, method, analyzed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    labels=excluded.labels, summary=excluded.summary,
                    severity=excluded.severity, confidence=excluded.confidence,
                    reason=excluded.reason, method=excluded.method,
                    analyzed_at=excluded.analyzed_at
                """,
                (
                    prediction.review_id, "|".join(prediction.labels), prediction.summary,
                    prediction.severity, prediction.confidence, prediction.reason,
                    prediction.method, now,
                ),
            )
        connection.execute(
            "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id, now, day, int(dry_run), provider_name, len(rows), len(new_rows),
                len(changed_rows), unchanged, ai_rows, elapsed, provider.usage.total_tokens,
            ),
        )
        connection.commit()
    connection.close()

    report_path = RESULTS_DIR / f"{'dry_run' if dry_run else 'ingest'}_day_{day}_{run_id}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    print(f"saved: {report_path.relative_to(ROOT)}")
    return report_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download")
    download_parser.add_argument("--force", action="store_true")

    subparsers.add_parser("prepare")

    benchmark_parser = subparsers.add_parser("benchmark")
    benchmark_parser.add_argument(
        "--provider", choices=("oracle", "openai", "gemini"), default="oracle"
    )
    benchmark_parser.add_argument("--sample-size", type=int, default=None)
    benchmark_parser.add_argument("--model", default=None)

    ingest_parser = subparsers.add_parser("ingest")
    ingest_parser.add_argument("--day", type=int, choices=range(1, 8), required=True)
    ingest_parser.add_argument("--dry-run", action="store_true")
    ingest_parser.add_argument(
        "--provider", choices=("oracle", "openai", "gemini"), default="oracle"
    )
    ingest_parser.add_argument("--model", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        download_data(force=args.force)
    elif args.command == "prepare":
        prepare_data()
    elif args.command == "benchmark":
        config = load_config()
        benchmark(
            provider_name=args.provider,
            sample_size=args.sample_size or int(config["benchmark_sample_size"]),
            model=args.model,
        )
    elif args.command == "ingest":
        ingest(
            day=args.day,
            dry_run=args.dry_run,
            provider_name=args.provider,
            model=args.model,
        )


if __name__ == "__main__":
    main()
