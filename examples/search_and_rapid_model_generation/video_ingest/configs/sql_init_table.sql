DROP TABLE IF EXISTS descriptor_index;
CREATE TABLE IF NOT EXISTS descriptor_index (
  uid       TEXT  NOT NULL,
  element   BYTEA NOT NULL,

  PRIMARY KEY (uid)
);
