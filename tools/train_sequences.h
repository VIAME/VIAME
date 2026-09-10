// This file is part of VIAME, distributed under the BSD 3-Clause License.
#ifndef VIAME_TOOLS_TRAIN_SEQUENCES_H
#define VIAME_TOOLS_TRAIN_SEQUENCES_H
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace viame { namespace tools {
struct sequence_frames
{
  std::vector<std::string> images;
  std::vector<size_t> first, count;
};

// Rebuild ranges from the final filtered partition, not pre-filter counts.
inline sequence_frames partition_sequences(
  const std::vector<std::vector<std::string>>& sources,
  const std::vector<std::string>& selected )
{
  std::map<std::string, size_t> owner;
  for( size_t i = 0; i < sources.size(); ++i )
  {
    for( const auto& file : sources[i] )
    {
      const auto entry = owner.emplace( file, i );
      if( !entry.second && entry.first->second != i )
      {
        throw std::runtime_error("Frame belongs to multiple training sequences: " + file);
      }
    }
  }
  std::vector<std::vector<std::string>> grouped(sources.size());
  for( const auto& file : selected )
  {
    auto it = owner.find(file);
    if( it == owner.end() )
    {
      throw std::runtime_error("Frame has no training sequence: " + file);
    }
    grouped[it->second].push_back(file);
  }
  sequence_frames result;
  for( const auto& group : grouped )
  {
    result.first.push_back(result.images.size());
    result.count.push_back(group.size());
    result.images.insert(result.images.end(), group.begin(), group.end());
  }
  return result;
}
} }
#endif
