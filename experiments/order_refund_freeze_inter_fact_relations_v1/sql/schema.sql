PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE orders(
  order_id TEXT PRIMARY KEY,
  amount_cents INTEGER NOT NULL,
  status TEXT NOT NULL,
  version INTEGER NOT NULL
);

CREATE TABLE refunds(
  refund_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL,
  amount_cents INTEGER NOT NULL,
  status TEXT NOT NULL,
  idempotency_key TEXT NOT NULL UNIQUE,
  FOREIGN KEY(order_id) REFERENCES orders(order_id)
);

CREATE TABLE notifications(
  notification_id TEXT PRIMARY KEY,
  order_id TEXT NOT NULL,
  refund_id TEXT,
  status TEXT NOT NULL,
  notification_kind TEXT NOT NULL,
  FOREIGN KEY(order_id) REFERENCES orders(order_id)
);
