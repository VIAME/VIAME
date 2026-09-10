"""Verify frame-to-sequence association independently of model training."""
import shutil
import subprocess
from pathlib import Path
import pytest


def test_filtered_sequence_partitions(tmp_path):
    compiler = shutil.which('c++')
    if not compiler:
        pytest.skip('C++ compiler not available')
    source = r'''
#include "train_sequences.h"
#include <cassert>
int main() {
  using viame::tools::partition_sequences;
  std::vector<std::vector<std::string>> items = {{"a0", "a1", "a2"}, {"b0", "b1"}};
  auto train = partition_sequences(items, {"a0", "a2"});
  auto val = partition_sequences(items, {"b0", "b1"});
  assert(train.count[0] == 2 && train.count[1] == 0);
  assert(val.count[0] == 0 && val.count[1] == 2 && val.first[1] == 0);
  auto filtered = partition_sequences(items, {"b1", "a2"});
  assert(filtered.images == std::vector<std::string>({"a2", "b1"}));
  assert(filtered.first[1] == 1);
  bool rejected = false;
  try { partition_sequences({{"same"}, {"same"}}, {"same"}); }
  catch(const std::runtime_error&) { rejected = true; }
  assert(rejected);
}
'''
    executable = tmp_path / 'partition-test'
    tools = Path(__file__).resolve().parents[2] / 'tools'
    subprocess.run([compiler, '-std=c++17', '-I', str(tools), '-x', 'c++', '-', '-o', str(executable)],
                   input=source, text=True, check=True, capture_output=True)
    subprocess.run([str(executable)], check=True)
